import numpy as np, cv2, imageio
import os
import map_and_plan_agent.depth_utils as du
import map_and_plan_agent.rotation_utils as ru
from map_and_plan_agent.fmm_planner import FMMPlanner
import skimage
import matplotlib.pyplot as plt
import subprocess as sp
import util
import math
import imutils

import numpy.ma as ma
import habitat_sim.utils.common as hutil
from matplotlib import colors
from matplotlib.collections import LineCollection


def subplot(plt, Y_X, sz_y_sz_x=(10, 10)):
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    plt.rcParams['figure.figsize'] = (X * sz_x, Y * sz_y)
    fig, axes = plt.subplots(Y, X)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig, axes


# make sure dt is 30
# step size is currently 25
# upper limit = height
class DepthMapperAndPlanner(object):
    def __init__(self,
                 dt=10,
                 camera_height=125.,
                 upper_lim=125.,
                 map_size_cm=6000,
                 out_dir=None,
                 mark_locs=False,
                 reset_if_drift=False,
                 count=-1,
                 close_small_openings=False,
                 recover_on_collision=False,
                 fix_thrashing=False,
                 goal_f=1.1,
                 point_cnt=2,
                 habitat_start_rot=None,
                 habitat_start_pos=None,
                 forward_step_size=0.25,
                 avd=False,
                 log_visualization=True):
        self.map_size_cm = map_size_cm
        self.dt = dt
        self.count = count
        self.out_dir = out_dir
        self.mark_locs = mark_locs
        self.reset_if_drift = reset_if_drift
        self.elevation = 0.  #np.rad2deg(env_config.SIMULATOR.DEPTH_SENSOR.ORIENTATION[0])
        self.camera_height = camera_height
        self.upper_lim = upper_lim
        # passed in in meters, converted to cm for internal use
        self.forward_step_size = forward_step_size * 100
        # because the navmeshes have maxclib of 20
        self.lower_lim = 20
        self.avd = avd
        # self.lower_lim = 1
        self.close_small_openings = close_small_openings
        self.num_erosions = 2
        self.recover_on_collision = recover_on_collision
        self.fix_thrashing = fix_thrashing
        self.goal_f = goal_f
        self.point_cnt = point_cnt
        self.log_visualization = log_visualization
        self.fmmMapCache = None
        # print(self.elevation, self.camera_height, self.upper_lim, self.lower_lim)

    def reset(self):
        self.RESET = True

    def _reset(self,
               goal_dist,
               start_pos,
               start_ang,
               soft=False,
               global_goals=[],
               camera_attrs=None):
        # Create an empty map of some size
        resolution = self.resolution = 5
        # self.selem = skimage.morphology.disk(15 / resolution)
        # self.selem_small = skimage.morphology.disk(1)
        self.selem = skimage.morphology.disk(1)
        # 0 agent moves forward. Agent moves in the direction of +x
        # 1 rotates left
        # 2 rotates right
        # 3 agent stop
        self.z_bins = [self.lower_lim, self.upper_lim]
        map_size_cm = np.maximum(self.map_size_cm,
                                 goal_dist * 2 * self.goal_f) // resolution
        map_size_cm = int(map_size_cm * resolution)
        self.map = np.zeros(
            (map_size_cm // resolution + 1, map_size_cm // resolution + 1,
             len(self.z_bins) + 1),
            dtype=np.float32)
        self.loc_on_map = np.zeros(
            (map_size_cm // resolution + 1, map_size_cm // resolution + 1),
            dtype=np.float32)
        self.current_loc = np.array([(self.map.shape[0] - 1) / 2,
                                     (self.map.shape[0] - 1) / 2, start_ang],
                                    np.float32)
        self.current_loc[:2] = self.current_loc[:2] * resolution
        self.start_loc = self.current_loc.copy()
        self.last_step_angle = self.current_loc[2]

        if camera_attrs is not None:
            self.camera = du.get_camera_matrix(*camera_attrs)
        else:
            self.camera = du.get_camera_matrix(224, 224, 90)
        self.goal_loc = None
        self.last_act = 3
        self.locs = []
        self.acts = []
        self.last_pointgoal = None
        self.trials = 0
        self.rgbs = []
        self.maps = []
        self.depths = []
        self.pans = []
        self.current_pan = None
        self.current_open = None
        self.reasoning_locs = []
        # self.global_goals = np.array([ self.xy_from_pointgoal(e) for e in global_goals ])
        self.start_pos = start_pos
        self.start_ang = start_ang
        self.global_goals = np.array([[self.pos_to_loc(e) for e in pts]
                                      for pts in global_goals])
        self.comitted_actions = None
        if not soft:
            self.num_resets = 0
            self.count = self.count + 1
            self.trials = 0
            self.rgbs = []
            self.depths = []
            self.recovery_actions = []
            self.thrashing_actions = []

    def add_observation(self, depth, loc=None, height=None):
        if loc is None: loc = self.current_loc
        if height is None:
            height = self.camera_height
        # depth is in cm
        d = depth[:, :, 0] * 1
        d[d > 990] = np.NaN
        d[d == 0] = np.NaN
        XYZ1 = du.get_point_cloud_from_z(d, self.camera)
        XYZ2 = du.make_geocentric(XYZ1 * 1, height, self.elevation)
        # Transform pose
        # Rotate and then translate by agent center
        XYZ3 = self.transform_to_current_frame(XYZ2, loc)
        counts, is_valids = du.bin_points(XYZ3, self.map.shape[0], self.z_bins,
                                          self.resolution)
        self.map = self.map + counts

        # invalidate map cache
        self.fmmMapCache = None

    def loc_to_map(self, loc):
        return np.flip((loc[:2] // self.resolution)).astype(np.int)

    # def angle_diff(self,a,b):
    # return math.atan2(math.sin(a-b),math.cos(a-b))

    def get_action_toward(self, pos, db=False):
        if db:
            import pdb
            pdb.set_trace()

        # if thrashing we commit to an action sequence and execute as long as we're
        # going toward the same goal, comitted actions are popped off in the loc_act
        # function

        if self.comitted_actions is not None and (
                self.comitted_actions[0] == np.array(pos)).all() and len(
                    self.comitted_actions[1]) > 0:
            return self.comitted_actions[1][0]
        else:
            self.comitted_actions = None

        traversible = self.get_traversible()
        distances = self.fmmMap(pos=pos)
        max_rots = 180 // self.dt

        # 2 step search
        def with_next_step(arr):
            ret = [arr + [0]]
            for i in range(1, max_rots + 1):
                ret += [arr + [1] * i + [0]]
                ret += [arr + [2] * i + [0]]
            return ret

        sequences = [[3]] + with_next_step([])

        # don't add steps with avd
        if not self.avd:
            for seq in with_next_step([]):
                sequences += with_next_step(seq)

        start_map_pos = self.loc_to_map(self.current_loc)

        # if distances[tuple(start_map_pos)] > 50/self.resolution:
        # sequences = [[0]]
        # else:
        # sequences = [[3],[0]]
        # for i in range(1,max_rots+1):
        # sequences += [[1]*i+[0]]
        # sequences += [[2]*i+[0]]

        rads = np.pi * self.dt / 180
        # in cms
        step_size = self.forward_step_size

        def score_sequence(seq):
            pos = self.current_loc[0:2]
            rot = self.current_loc[2]
            stepped = False
            for a in seq:
                if a == 1: rot += rads
                if a == 2: rot -= rads
                if a == 0:
                    disp = np.array([math.cos(rot), math.sin(rot)]) * step_size

                    # moving back in the direction you came is always traversible
                    # if abs(self.angle_diff(rot,self.last_step_angle+np.pi)) > np.pi/12 or stepped:
                    for prop in np.linspace(0, 1, num=10):
                        map_pos = self.loc_to_map(disp * prop + pos)
                        if not traversible[map_pos[0], map_pos[1]]:
                            return (1, disp + pos)
                    pos = disp + pos
                    stepped = True

            map_pos = self.loc_to_map(pos)
            # print(map_pos)
            return (distances[tuple(map_pos)] -
                    distances[tuple(start_map_pos)] + len(seq) * 0.1, map_pos)

        # import pdb; pdb.set_trace()
        # score_sequence()

        ####

        # dist = distances.copy()
        # map_pos = self.loc_to_map(self.current_loc)
        # dist[tuple(map_pos)] = 500
        # dist[tuple(score_sequence(item)[1])] = 400
        # plt.imsave('vis/test.png', dist)

        # plt.imsave('vis/test.png',self.get_map_rgb())
        ####3
        ind, item, val = util.argmin(sequences, lambda x: score_sequence(x)[0])
        if self.avd and (item[0] == 1 and self.last_act == 2) or (
                item[0] == 2 and self.last_act == 1):
            self.comitted_actions = (pos, item)
        return item[0]

    def get_best_action(self):
        None

    def xy_from_pointgoal(self, pointgoal):
        xy = self.compute_xy_from_pointnav(pointgoal)
        res = xy * 1
        res[0] = res[0] + self.current_loc[0]
        res[1] = res[1] + self.current_loc[1]
        return res

    # def set_goal(self,pointgoal):
    # print('setgoal',pointgoal)
    # self.goal_loc = self.xy_from_pointgoal(pointgoal)

    def set_goal(self, pos):
        self.goal_loc = self.pos_to_loc(pos)

    def transform_to_current_frame(self, XYZ, loc=None):
        if loc is None: loc = self.current_loc
        R = ru.get_r_matrix([0., 0., 1.], angle=loc[2] - np.pi / 2.)
        XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
        XYZ[:, :, 0] = XYZ[:, :, 0] + loc[0]
        XYZ[:, :, 1] = XYZ[:, :, 1] + loc[1]
        return XYZ

    def pos_to_loc(self, pos):
        diff = pos - self.start_pos
        displacement = np.array([-diff[2], -diff[0]]) * 100
        return displacement + self.start_loc[:2]

    def new_update_loc(self, pos, ang):
        prev_loc = self.current_loc.copy()
        diff = pos - self.start_pos
        displacement = np.array([-diff[2], -diff[0]]) * 100
        self.current_loc[:2] = displacement + self.start_loc[:2]
        self.current_loc[2] = ang
        self.locs.append(self.current_loc + 0)
        self.mark_on_map(self.current_loc)

        # if the location moved (not just rotation) save the previous location
        if not all(prev_loc[:2] == self.current_loc[:2]):
            self.last_step_angle == self.current_loc[2]

    def update_loc(self, last_act, pointgoal=None):
        # Currently ignores goal_loc.
        if last_act == 1:
            self.current_loc[2] = self.current_loc[2] + self.dt * np.pi / 180.
        elif last_act == 2:
            self.current_loc[2] = self.current_loc[2] - self.dt * np.pi / 180.
        elif last_act == 0:
            self.current_loc[0] = self.current_loc[
                0] + self.forward_step_size * np.cos(self.current_loc[2])
            self.current_loc[1] = self.current_loc[
                1] + self.forward_step_size * np.sin(self.current_loc[2])
        self.locs.append(self.current_loc + 0)
        self.mark_on_map(self.current_loc)

    def mark_on_map(self, loc):
        x = int(loc[0] // self.resolution)
        y = int(loc[1] // self.resolution)
        self.loc_on_map[y, x] = 1

    def point_goal_to_global_xy(self, pointgoal):
        xy = self.compute_xy_from_pointnav(pointgoal)
        goal = xy * 1
        goal[0] = goal[0] + self.current_loc[0]
        goal[1] = goal[1] + self.current_loc[1]
        return goal

    def log_reasoning(self):
        self.reasoning_locs.append(self.current_loc.copy())

    def save_vis(self, extras=[]):
        if self.trials < 20:
            fig, axes = subplot(plt, (1, 3))
            axes = axes.ravel()[::-1].tolist()
            ax = axes.pop()

            locs = np.array(self.locs).reshape([-1, 3])
            acts = np.array(self.acts).reshape([-1])
            ax.imshow(self.map[:, :, 1] > 0, origin='lower')

            ax.plot(locs[:, 0] / 5, locs[:, 1] / 5, 'm.', ms=3)
            if locs.shape[0] > 0:
                ax.plot(locs[0, 0] / 5, locs[0, 1] / 5, 'bx')
            ax.plot(self.current_loc[0] / 5, self.current_loc[1] / 5, 'b.')
            ax.plot(self.goal_loc[0] / 5, self.goal_loc[1] / 5, 'y*')

            for g in self.global_goals:
                ax.plot(g[0] / 5, g[1] / 5, 'y*')

            for e in extras:
                g = self.point_goal_to_global_xy(e)
                ax.plot(g[0] / 5, g[1] / 5, 'b*')

            ax = axes.pop()
            ax.imshow(self.fmm_dist, origin='lower')
            ax.plot(locs[:, 0] / 5, locs[:, 1] / 5, 'm.', ms=3)
            if locs.shape[0] > 0:
                ax.plot(locs[0, 0] / 5, locs[0, 1] / 5, 'bx')
            ax.plot(self.current_loc[0] / 5, self.current_loc[1] / 5, 'b.')
            ax.plot(self.goal_loc[0] / 5, self.goal_loc[1] / 5, 'y*')

            ax = axes.pop()
            ax.plot(acts)
            plt.savefig(os.path.join(
                self.out_dir,
                '{:04d}_{:03d}.png'.format(self.count, self.trials)),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.out_dir,
                                     '{:04d}.png'.format(self.count)),
                        bbox_inches='tight')
            plt.close()

    def soft_reset(self, pointgoal):
        # This reset is called if there is drift in the position of the goal
        # location, indicating that there had been collisions.
        if self.out_dir is not None:
            self.save_vis()
        self._reset(pointgoal[0] * 100., soft=True)
        self.trials = self.trials + 1
        self.num_resets = self.num_resets + 1
        xy = self.compute_xy_from_pointnav(pointgoal)
        # self.current_loc has been set inside reset
        self.goal_loc = xy * 1
        self.goal_loc[0] = self.goal_loc[0] + self.current_loc[0]
        self.goal_loc[1] = self.goal_loc[1] + self.current_loc[1]
        self.mark_on_map(self.goal_loc)
        self.mark_on_map(self.current_loc)
        if self.num_resets == 6:
            # We don't want to keep resetting. First few resets fix themselves,
            # so do it for later resets.
            num_rots = int(np.round(180 / self.dt))
            self.recovery_actions = [1] * num_rots + [0] * 6
        else:
            self.recovery_actions = []

    def check_drift(self, pointgoal):
        goal_loc = self.xy_from_pointgoal(pointgoal)
        return np.linalg.norm(goal_loc - self.goal_loc) > 5

    def check_thrashing(self, n, acts):
        thrashing = False
        if len(acts) > n:
            last_act = acts[-1]
            thrashing = last_act == 1 or last_act == 2
            for i in range(2, n + 1):
                if thrashing:
                    thrashing = acts[-i] == 3 - last_act
                    last_act = acts[-i]
                else:
                    break
        return thrashing

    def compute_xy_from_pointnav(self, pointgoal):
        xy = np.array([
            np.cos(pointgoal[1] + self.current_loc[2]),
            np.sin(pointgoal[1] + self.current_loc[2])
        ],
                      dtype=np.float32)
        xy = xy * pointgoal[0] * 100
        return xy

    def point_goal_to_global_xy(self, pointgoal):
        xy = self.compute_xy_from_pointnav(pointgoal)
        goal = xy * 1
        goal[0] = goal[0] + self.current_loc[0]
        goal[1] = goal[1] + self.current_loc[1]
        return goal

    # logs an action take and updates the position
    # action should be the action just taken and everthing
    # else is the current obs,pos,ang
    def log_act(self, obs, pos, ang, action):
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        if len(depth.shape) == 4: depth = depth[0]
        if len(rgb.shape) == 4: rgb = rgb[0]
        old_loc = self.current_loc.copy()
        self.new_update_loc(pos, ang)

        # invalidate map cache
        self.fmmMapCache = None
        if 'pose' in obs:
            # for differences in height in avd
            self.add_observation(depth * 1000, height=obs['pose'][1])
        else:
            self.add_observation(depth * 1000)

        if action == 0:
            dist = np.linalg.norm((self.current_loc - old_loc)[:2])
            # check for collisions and place map points infront of the agnet
            if (self.avd and dist < 1e-3) or (not self.avd and dist <= 24):
                print("Collision detected")
                if self.avd:
                    collision_radius = np.pi / 8
                    block_range = range(int(self.forward_step_size - 3),
                                        int(self.forward_step_size))
                else:
                    collision_radius = np.pi / 6
                    block_range = range(10, 15)

                angles = np.linspace(-collision_radius / 2,
                                     collision_radius / 2,
                                     num=25)
                for block_dist in block_range:
                    for angle_offset in angles:
                        object_loc = self.current_loc[:2].copy()
                        object_loc[0] += block_dist * np.cos(
                            self.current_loc[2] + angle_offset)
                        object_loc[1] += block_dist * np.sin(
                            self.current_loc[2] + angle_offset)
                        obj_map = self.loc_to_map(object_loc)
                        self.map[obj_map[0], obj_map[1], 1] += self.point_cnt

        # should never fire
        # drift = self.check_drift(pointgoal)
        # if drift:
        # print("drift_detected")

        if self.comitted_actions is not None:
            if action == self.comitted_actions[1][0]:
                self.comitted_actions[1].pop(0)
            else:
                raise Exception(f'bad action')
        self.last_act = action
        self.acts.append(action)
        if self.log_visualization:
            frames = 2 if action in [1, 2] else 1
            for _ in range(frames):
                if not depth.shape[0:2] == (224, 224):
                    resized_depth = imutils.resize(depth, height=224)
                    start = resized_depth.shape[1] // 2 - 112
                    resized_depth = resized_depth[:, start:start + 224]
                    resized_depth[resized_depth > 1] = 1
                    self.depths.append(
                        (resized_depth[..., 0] * 255).astype(np.uint8))
                else:
                    self.depths.append((depth[..., 0] * 255).astype(np.uint8))
                self.rgbs.append(rgb)
                self.maps.append(self.get_map_rgb())
                self.pans.append(self.current_pan)

    # def act(self, obs, pos,ang):
    # rgb = obs['rgb'].astype(np.uint8)
    # depth = obs['depth']
    # if len(depth.shape) == 4: depth=depth[0]
    # if len(rgb.shape) == 4: rgb=rgb[0]
    # pointgoal = obs['pointgoal']
    # # self.update_loc(self.last_act)
    # old_loc = self.current_loc.copy()
    # self.new_update_loc(pos,ang)

    # if self.last_act == 0:
    # dist =  np.linalg.norm((self.current_loc-old_loc)[:2])
    # # check for collisions
    # # and place map points infront of the agnet
    # if dist <= 24:
    # print("Collision detected")
    # # import pdb; pdb.set_trace()
    # # plt.imsave('vis/test/map.png',self.get_map_rgb())
    # # plt.imsave('vis/test/map.png',self.map[:,:,1] >= self.point_cnt)
    # # plt.imsave('vis/test/map.png',self.map[:,:,1])
    # # block_dist = 20
    # collision_radius = np.pi/6
    # angles = np.linspace(-collision_radius/2,collision_radius/2,num=25)
    # for block_dist in range(10,15):
    # for angle_offset in angles:
    # object_loc = self.current_loc[:2].copy()
    # object_loc[0] += block_dist*np.cos(self.current_loc[2]+angle_offset)
    # object_loc[1] += block_dist*np.sin(self.current_loc[2]+angle_offset)
    # obj_map = self.loc_to_map(object_loc)
    # self.map[obj_map[0],obj_map[1],1] += self.point_cnt

    # drift = self.check_drift(pointgoal)
    # if drift:
    # print("drift_detected")
    # self.xy_from_pointgoal(pointgoal)
    # self.goal_loc
    # import pdb; pdb.set_trace()
    # if self.reset_if_drift and drift:
    # self.soft_reset(pointgoal)

    # self.add_observation(depth*1000)
    # act = self.get_action_toward(self.goal_loc)

    # self.acts.append(act)
    # self.last_act = act

    # # slowdown framerate of turns
    # frames = 2 if act in [1,2] else 1
    # # double check input is [0,1]
    # for _ in range(frames):
    # self.depths.append((depth[...,0]*255).astype(np.uint8))
    # self.rgbs.append(rgb)
    # self.maps.append(self.get_map_rgb())
    # self.pans.append(self.current_pan)

    # self.last_pointgoal = pointgoal + 0

    # return act

    # checks if there would be an action other than stop
    def action_toward(self, goal_pos):
        # act, _ = self.plan_path(self.pos_to_loc(goal_pos))
        act = self.get_action_toward(goal_pos)
        return act != 3

    def fmmMap(self, pos=None, loc=None, close=True):
        if pos is not None:
            goal_loc = self.pos_to_loc(pos)
        elif loc is not None:
            goal_loc = loc
        else:
            goal_loc = self.current_loc
        map_loc = (goal_loc.astype(np.int32) // self.resolution)[:2]

        # same location and cache valid
        if self.fmmMapCache is not None and (
                map_loc == self.fmmMapCache[0]).all():
            return self.fmmMapCache[1]

        traversible = self.get_traversible()
        if self.close_small_openings and close:
            n = self.num_erosions
            reachable = False
            while n >= 0 and not reachable:
                traversible_open = traversible.copy()
                for i in range(n):
                    traversible_open = skimage.morphology.binary_erosion(
                        traversible_open, self.selem)
                for i in range(n):
                    traversible_open = skimage.morphology.binary_dilation(
                        traversible_open, self.selem)
                planner = FMMPlanner(traversible_open, 360 // self.dt)
                dists = planner.distances(map_loc)
                cur_map_loc = self.loc_to_map(self.current_loc)
                reachable = dists[tuple(cur_map_loc)] != np.inf
                n = n - 1
        else:
            planner = FMMPlanner(traversible, 360 // self.dt)
            dists = planner.distances(map_loc)

        self.fmmMapCache = (map_loc, dists)
        return dists

    #Returns the distance reported by the fmm scaled to meters
    def fmmDistance(self, point):
        dists = self.fmmMap(pos=point)
        return dists[tuple(self.loc_to_map(
            self.current_loc))] * self.resolution / 100

    # def fmmDistances(self,points):
    # dists = self.fmmMap()
    # point_coords = np.array([ self.loc_to_map(self.pos_to_loc(e)) for e in points ])

    # mask = np.logical_or(point_coords[:,0] > dists.shape[0],point_coords[:,1] > dists.shape[1])
    # point_coords[:,0][mask] = 0
    # point_coords[:,1][mask] = 0

    # output =  dists[point_coords[:,0],point_coords[:,1]]*self.resolution/100
    # output[mask] = np.inf
    # return output

    def get_traversible(self):
        loc = self.loc_to_map(self.current_loc)[:2]
        obstacle = self.map[:, :, 1] >= self.point_cnt
        if self.mark_locs:
            obstacle[loc[0], loc[1]] = False
        traversible = skimage.morphology.binary_dilation(obstacle,
                                                         self.selem) != True
        traversible[loc[0], loc[1]] = True
        return traversible

    def reachable_nearby(self, points, debug=False):
        # coppied from planpath
        # traversible = self.get_traversible()
        # planner = FMMPlanner(traversible, 360 // self.dt)
        # loc = (self.current_loc.astype(np.int32) // self.resolution)[:2]
        # dists = planner.distances(loc).transpose()
        dists = self.fmmMap(loc=self.current_loc, close=True).transpose()

        # grid_points = np.array([self.point_goal_to_global_xy(p)//self.resolution for p in points]).astype(int)
        grid_points = np.array([
            self.pos_to_loc(p) // self.resolution for p in points
        ]).astype(int)

        # output =  dists[point_coords[:,0],point_coords[:,1]]*self.resolution/100

        # dont allow samples outside the map
        # grid_points[:,0] = np.clip(grid_points[:,0],0,dists.shape[0]-1)
        # grid_points[:,1] = np.clip(grid_points[:,1],0,dists.shape[1]-1)

        # if debug:
        # import pdb; pdb.set_trace()

        # self.current_loc//self.resolution
        # dist_arr[grid_points[:,0],grid_points[:,1]] = 0
        # dist_arr[328,284]
        # plt.clf()
        # plt.imshow(dists)
        # plt.savefig(f'{self.out_dir}/dd.png')

        # plt.imsave(f'{self.out_dir}/dists.png',dist_arr)
        # plt.imsave(f'{self.out_dir}/map.png',obstacle)
        # self.save_vis(points)
        # dists[grid_points[:,0],grid_points[:,1]]
        # dists

        # dont allow samples outside the map
        mask = np.logical_or(grid_points[:, 0] >= dists.shape[0],
                             grid_points[:, 1] >= dists.shape[1])
        mask = np.logical_or(mask, grid_points[:, 0] < 0,
                             grid_points[:, 1] < 0)
        grid_points[:, 0][mask] = 0
        grid_points[:, 1][mask] = 0

        # distance to each point in meters
        point_dists = dists[grid_points[:, 0],
                            grid_points[:, 1]] * self.resolution / 100
        point_dists[mask] = np.inf

        if np.sum(point_dists < 3) > 0:
            return np.argmax(point_dists < 3)
        else:
            return None

    def write_mp4_imageio(self):
        sz = self.rgbs[0].shape[0]
        out_file_name = os.path.join(self.out_dir,
                                     '{:04d}.gif'.format(self.count))
        imageio.mimsave(out_file_name, self.rgbs)

        sz = self.depths[0].shape[0]
        out_file_name = os.path.join(self.out_dir,
                                     '{:04d}_d.gif'.format(self.count))
        imageio.mimsave(out_file_name, self.depths)

    def get_map_rgb(self, orig_range=None):
        marker_size = 15
        crop_shift = np.array([0, 0])
        crop_range = None
        if orig_range is not None:
            marker_size = 30
            # hacky scaling
            crop_range = ((np.array(orig_range) * self.map.shape[0]) / 1000).astype(np.int)
            flipped_range = self.map.shape[0]-crop_range[0]
            crop_range[0] = [flipped_range[1],flipped_range[0]]
            crop_shift = np.array([crop_range[1][0], crop_range[0][0]])
        map_to_render = (self.map[:, :, 1] > self.point_cnt).astype(np.uint8)
        if crop_range is not None:
            map_to_render = map_to_render[crop_range[0][0]:crop_range[0][1],
                                          crop_range[1][0]:crop_range[1][1]]

        fig, ax = subplot(plt, (1, 1))
        locs = np.array(self.locs).reshape([-1, 3])[:, :2] / 5
        locs -= crop_shift
        lightGreen = colors.ListedColormap(['white', '#35a655'])

        ax.imshow(map_to_render,
                  origin='lower',
                  aspect='auto',
                  cmap=lightGreen,
                  vmin=0,
                  vmax=1)
        ax.plot(locs[:, 0], locs[:, 1], 'k.', ms=marker_size / 2)
        if locs.shape[0] > 0:
            ax.plot(locs[0, 0], locs[0, 1], 'kx', ms=marker_size / 2)

        # bigger dot on reasoning steps
        reas = np.stack(self.reasoning_locs)[:, :-1] / 5 - crop_shift
        ax.plot(reas[:, 0], reas[:, 1], 'k.', ms=marker_size)

        if self.current_open:
            open_locs = np.stack(
                list(map(lambda x: self.pos_to_loc(x[1]),
                         self.current_open))) / 5 - crop_shift
            ax.plot(open_locs[:, 0],
                    open_locs[:, 1],
                    color='#1ca4fc',
                    linestyle='none',
                    marker='.',
                    ms=marker_size)

        goal_loc = self.goal_loc / 5 - crop_shift
        ax.plot(goal_loc[0],
                goal_loc[1],
                color='#862117',
                linestyle='none',
                marker='.',
                ms=marker_size)
        lines = []
        for obj in ((self.global_goals / 5) - crop_shift):
            lines += [(obj[i], obj[(i + 1) % len(obj)])
                      for i in range(len(obj))]
            ax.add_collection(LineCollection(lines, linewidth=4, color='r',clip_on=True))

        current_loc = self.current_loc[:2] / 5 - crop_shift
        ax.plot(current_loc[0],
                current_loc[1],
                'k.',
                linestyle='none',
                ms=marker_size)

        disp = np.array(
            [math.cos(self.current_loc[2]),
             math.sin(self.current_loc[2])]) * 10
        ax.arrow(current_loc[0],
                 current_loc[1],
                 disp[0],
                 disp[1],
                 head_width=4,
                 head_length=4,
                 fc='r',
                 ec='r')
        ax.set_axis_off()
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.canvas.draw()
        ax.get_window_extent()
        ax.set_xlim((0,map_to_render.shape[1]))
        ax.set_ylim((0,map_to_render.shape[0]))
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        # plt.imsave('vis/test.png',data)
        # fig.savefig('vis/test.png',bbox_inches='tight', pad_inches=0)
        if orig_range is not None:
            return fig
        else:
            plt.close(fig)
            return data

    def set_current_pan(self, pan):
        self.current_pan = pan

    def set_current_open(self, op):
        self.current_open = op

    def write_combined(self, suffix="", class_text="", maptitle=''):
        res = []
        size = (self.map.shape[0] - 1) * self.resolution
        locs = np.array(self.locs)
        # transpose locs?
        locs = locs[:, (1, 0)]
        locs[:, 0] = size - locs[:, 0]
        mins = np.min(locs, axis=0)
        maxs = np.max(locs, axis=0)
        mins = mins / self.resolution - 25
        maxs = maxs / self.resolution + 25
        scale = 1000 / self.map[0].shape[0]
        mins = (mins * scale).astype(np.int)
        maxs = (maxs * scale).astype(np.int)

        # import pdb; pdb.set_trace()

        # make the map square
        diffs = ((maxs - mins).max() - (maxs - mins)) // 2
        maxs += diffs
        mins -= diffs

        for rgb, depth, map_rgb, pan in zip(self.rgbs, self.depths, self.maps,
                                            self.pans):
            color_depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :-1]

            map_rgb = map_rgb[mins[0]:maxs[0], mins[1]:maxs[1], :]
            if pan is None:
                pan = np.zeros_like(self.pans[-1])

            scale = rgb.shape[0] / map_rgb.shape[0]
            shaped_map = cv2.resize(map_rgb, (0, 0), fx=scale, fy=scale)
            plan_frame = np.concatenate((rgb, color_depth, shaped_map), axis=1)
            scale = plan_frame.shape[1] / self.current_pan.shape[1]
            pan_to_write = cv2.resize(pan, (0, 0), fx=scale, fy=scale)
            # text_pan = pan_to_write.copy()
            # text = f"Object Class: {class_text}"
            # (width, height), _ = cv2.getTextSize(text,
            # cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            # 1)
            # cv2.putText(text_pan, text, (text_pan.shape[1] - width - 5,
            # text_pan.shape[0] - height + 2),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            out = np.concatenate((plan_frame, pan_to_write),
                                 axis=0).astype(np.uint8)
            res.append(out)

        out_file_name = os.path.join(self.out_dir, f'slam{suffix}.mp4')
        imageio.mimsave(out_file_name, res)
        final_map = self.maps[-1][mins[0]:maxs[0], mins[1]:maxs[1], :]
        # import pdb; pdb.set_trace()
        final_map = self.get_map_rgb(((mins[0], maxs[0]), (mins[1], maxs[1])))
        final_map.axes[0].set_title(maptitle, fontsize=30, pad=15)
        final_map.axes[0].axis('off')
        path = os.path.join(self.out_dir, f'slam{suffix}.pdf')
        final_map.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(final_map)

    def write_mp4_cv2(self):
        sz = self.rgbs[0].shape[0]
        out_file_name = os.path.join(self.out_dir,
                                     '{:04d}.mp4'.format(self.count))
        video = cv2.VideoWriter(out_file_name, -1, 10, (sz, sz))
        for rgb in self.rgbs:
            video.write(rgb[:, :, ::-1])
        video.release()

    def write_mp4(self):
        sz = self.depths[0].shape[0]
        out_file_name = os.path.join(self.out_dir,
                                     '{:04d}.mp4'.format(self.count))
        ffmpeg_bin = 'ffmpeg'
        command = [
            ffmpeg_bin,
            '-y',  # (optional) overwrite output file if it exists
            '-f',
            'rawvideo',
            '-vcodec',
            'rawvideo',
            '-s',
            '{:d}x{:d}'.format(sz, sz),  # size of one frame
            '-pix_fmt',
            'rgb24',
            '-r',
            '4',  # frames per second
            '-i',
            '-',  # The imput comes from a pipe
            '-an',  # Tells FFMPEG not to expect any audio
            '-vcodec',
            'mpeg',
            out_file_name
        ]
        pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
        for rgb in self.rgbs:
            pipe.proc.stdin.write(rgb.tostring())
        # self.add_observation(depth, goal_vec, self.last_act)
        # act = get_action(self.map)
        # self.last_act = act
