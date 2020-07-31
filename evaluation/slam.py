import numpy as np, cv2, imageio
import os
import map_and_plan_agent.depth_utils as du
import map_and_plan_agent.rotation_utils as ru
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
from skimage.morphology import binary_closing, disk
import scipy, skfmm


class FMMPlanner():
    def __init__(self, traversible):
        self.traversible = traversible

    def distances(self, goal):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0]), int(goal[1])
        if goal_y >= traversible_ma.shape[0] or goal_x >= traversible_ma.shape[
                1] or goal_y < 0 or goal_x < 0:
            return np.ones_like(traversible_ma) * np.inf
        traversible_ma[goal_y, goal_x] = 0
        return skfmm.distance(traversible_ma, dx=1)


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
                 log_visualization=True):
        self.map_size_cm = map_size_cm
        self.dt = dt
        self.count = count
        self.out_dir = out_dir
        self.mark_locs = mark_locs
        self.reset_if_drift = reset_if_drift
        self.elevation = 0.
        self.camera_height = camera_height
        self.upper_lim = upper_lim
        # passed in in meters, converted to cm for internal use
        self.forward_step_size = forward_step_size * 100
        # because the navmeshes have maxclib of 20
        self.lower_lim = 20
        # self.lower_lim = 1
        self.close_small_openings = close_small_openings
        self.num_erosions = 2
        self.recover_on_collision = recover_on_collision
        self.fix_thrashing = fix_thrashing
        self.goal_f = goal_f
        self.point_cnt = point_cnt
        self.log_visualization = log_visualization
        self.fmmMapCache = None

    def _reset(self,
               goal_dist,
               start_pos,
               start_ang,
               soft=False,
               global_goals=[],
               camera_attrs=None):
        # Create an empty map of some size
        resolution = self.resolution = 5
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
        self.current_loc = np.array([(self.map.shape[0] - 1) / 2,
                                     (self.map.shape[0] - 1) / 2, start_ang],
                                    np.float32)
        self.current_loc[:2] = self.current_loc[:2] * resolution
        self.start_loc = self.current_loc.copy()

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

    def get_action_toward(self, pos, db=False):
        # if thrashing we commit to an action sequence and execute as long as we're
        # going toward the same goal, comitted actions are popped off in the loc_act
        # function
        if self.comitted_actions is not None and (
                self.comitted_actions[0] == pos).all() and len(
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

        for seq in with_next_step([]):
            sequences += with_next_step(seq)

        start_map_pos = self.loc_to_map(self.current_loc)
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

                    for prop in np.linspace(0, 1, num=10):
                        map_pos = self.loc_to_map(disp * prop + pos)
                        if not traversible[map_pos[0], map_pos[1]]:
                            return (1, disp + pos)
                    pos = disp + pos
                    stepped = True

            map_pos = self.loc_to_map(pos)
            return (distances[tuple(map_pos)] -
                    distances[tuple(start_map_pos)] + len(seq) * 0.1, map_pos)

        ind, item, val = util.argmin(sequences, lambda x: score_sequence(x)[0])
        return item[0]

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

    def log_reasoning(self):
        self.reasoning_locs.append(self.current_loc.copy())

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
        self.add_observation(depth * 1000)

        if action == 0:
            dist = np.linalg.norm((self.current_loc - old_loc)[:2])
            # check for collisions and place map points infront of the agnet
            if dist <= 24:
                print("Collision detected")
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

    # checks if there would be an action other than stop
    def action_toward(self, goal_pos):
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
                planner = FMMPlanner(traversible_open)
                dists = planner.distances(map_loc)
                cur_map_loc = self.loc_to_map(self.current_loc)
                reachable = dists[tuple(cur_map_loc)] != np.inf
                n = n - 1
        else:
            planner = FMMPlanner(traversible)
            dists = planner.distances(map_loc)

        self.fmmMapCache = (map_loc, dists)
        return dists

    #Returns the distance reported by the fmm scaled to meters
    def fmmDistance(self, point):
        dists = self.fmmMap(pos=point)
        return dists[tuple(self.loc_to_map(
            self.current_loc))] * self.resolution / 100

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
        dists = self.fmmMap(loc=self.current_loc, close=True).transpose()

        grid_points = np.array([
            self.pos_to_loc(p) // self.resolution for p in points
        ]).astype(int)

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

    def get_map_rgb(self):
        marker_size = 15
        fig, ax = subplot(plt, (1, 1))
        locs = np.array(self.locs).reshape([-1, 3])
        lightGreen = colors.ListedColormap(['white', '#35a655'])
        ax.imshow((self.map[:, :, 1] > self.point_cnt).astype(np.uint8),
                  origin='lower',
                  aspect='auto',
                  cmap=lightGreen,
                  vmin=0,
                  vmax=1)
        ax.plot(locs[:, 0] / 5, locs[:, 1] / 5, 'k.', ms=marker_size / 2)
        if locs.shape[0] > 0:
            ax.plot(locs[0, 0] / 5, locs[0, 1] / 5, 'kx', ms=marker_size / 2)

        # bigger dot on reasoning steps
        reas = np.stack(self.reasoning_locs)[:, :-1]
        ax.plot(reas[:, 0] / 5, reas[:, 1] / 5, 'k.', ms=marker_size)

        if self.current_open:
            open_locs = np.stack(
                list(map(lambda x: self.pos_to_loc(x[1]), self.current_open)))
            ax.plot(open_locs[:, 0] / 5,
                    open_locs[:, 1] / 5,
                    color='#1ca4fc',
                    linestyle='none',
                    marker='.',
                    ms=marker_size)

        # mark global goals
        ax.plot(self.goal_loc[0] / 5,
                self.goal_loc[1] / 5,
                color='#862117',
                linestyle='none',
                marker='.',
                ms=marker_size)
        lines = []
        for obj in self.global_goals / 5:
            lines += [(obj[i], obj[(i + 1) % len(obj)])
                      for i in range(len(obj))]

        ax.add_collection(LineCollection(lines, linewidth=2, color='r'))
        ax.plot(self.current_loc[0] / 5,
                self.current_loc[1] / 5,
                'k.',
                linestyle='none',
                ms=marker_size)

        cur_plot_pos = self.current_loc[:2] / 5
        disp = np.array(
            [math.cos(self.current_loc[2]),
             math.sin(self.current_loc[2])]) * 10
        ax.arrow(cur_plot_pos[0],
                 cur_plot_pos[1],
                 disp[0],
                 disp[1],
                 head_width=4,
                 head_length=4,
                 fc='r',
                 ec='r')

        ax.set_axis_off()
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close(fig)
        return data

    def set_current_pan(self, pan):
        self.current_pan = pan

    def set_current_open(self, op):
        self.current_open = op

    def write_combined(self, suffix="", class_text=""):
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
            out = np.concatenate((plan_frame, pan_to_write),
                                 axis=0).astype(np.uint8)
            res.append(out)

        out_file_name = os.path.join(self.out_dir, f'slam{suffix}.mp4')
        imageio.mimsave(out_file_name, res)
        final_map = self.maps[-1][mins[0]:maxs[0], mins[1]:maxs[1], :]
        imageio.imsave(os.path.join(self.out_dir, f'slam{suffix}.png'),
                       final_map)
