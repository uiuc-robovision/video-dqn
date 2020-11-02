# core dume without this torch import for some reason
import torch
import os
import numpy as np
from collections import Counter

import habitat
from habitat.utils.visualizations import maps
import cv2
from matplotlib import pyplot as plt
import util.habitat
import util.torch
import re
import util.cv2
from util.plt import show
import habitat_sim.utils.common as hutil
from time import sleep, time
from collections import namedtuple
import random
import traceback
import torchvision.transforms as transforms
import habitat_sim
from configs.habitat_config import get_config


class HabitatTestEnv:
    def __init__(self,
                 scene_location,
                 torchmode=True,
                 torch_notrans=False,
                 dest_index=None,
                 panorama=False,gpu_device_id=0,
                 config_path="configs/tasks/pointnav.yaml",goals=[],random_goal=False,turn_angle=30,num_floors=None,depth=False,rgb=True,semantics=False,imagenet_mode=False,allow_stairs=True,crop_to_square=False):

        config = get_config(config_paths=config_path)
        self.env = habitat.Env(config=config)
        self.dest_index = dest_index
        self.panorama = panorama
        self.random_goal = random_goal
        self.num_floors = num_floors
        self.imagenet_mode = imagenet_mode
        self.allow_stairs = allow_stairs
        self.crop_to_square = crop_to_square

        # for some reason doesn't work if I set the scene before first initilization
        config.defrost()
        config.SIMULATOR.SCENE = scene_location
        sensors = []
        if rgb: sensors.append('RGB_SENSOR')
        if depth: sensors.append('DEPTH_SENSOR')
        if semantics: sensors.append('SEMANTICS_SENSOR')
        config.SIMULATOR.SENSORS = sensors
        print(config.SIMULATOR.SENSORS)

        # config.SIMULATOR.DEPTH_SENSOR
        # this seems to not work, not sure why
        # im just changing it in the config file
        config.SIMULATOR.RGB_SENSOR.HEIGHT = 224
        config.SIMULATOR.RGB_SENSOR.WIDTH = 224
        config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 224
        config.SIMULATOR.DEPTH_SENSOR.WIDTH = 224
        config.SIMULATOR.TURN_ANGLE = turn_angle

        if habitat_sim.cuda_enabled:
            print("Using habitat with cuda")
            # fix device id
            config.SIMULATOR.HABITAT_SIM_V0.merge_from_list(
                ['GPU_DEVICE_ID', gpu_device_id,
                 "GPU_GPU", ((torchmode or torch_notrans) and (gpu_device_id is not None))
                ])

        config.freeze()
        self.env.sim.reconfigure(config.SIMULATOR)
        self.torchmode = torchmode
        self.torch_notrans = torch_notrans
        self.steps = 0
        self.goals = goals
        # if not random_goal and len(goals) == 0:
            # raise Exception('no goal specified')

        if len(goals) > 0:
            points = [self.env.sim.sample_navigable_point() for _ in range(1000)]
            if all([self._dist_to_goal(p) == float('inf') for p in points]):
                raise Exception('goals not reachable')

        # if not self.random_goal:
            # self.level = self.env.sim._sim.semantic_scene.levels[0]
            # self.bathrooms = list(
                # filter(lambda x: x.category.name() == 'bathroom',
                       # self.level.regions))

        #need to find where the first level is, because you can sample on the stairs
        # all I can think to do is sample to find the right heights for each level
        points = np.array([self.env.sim.sample_navigable_point() for _ in range(10000)])
        if self.num_floors:
            counts = Counter(points[:,1]).most_common(self.num_floors)
            self.floor_heights = sorted(map(lambda x: x[0],counts))
            

        # def path_to_any_goal(p):
            # return any([self.env.sim.geodesic_distance(p , g) != float('inf') for g in goals])

        # if len(goals) > 0:
            # points = np.stack([p for p in points if path_to_any_goal(p)])
        # self.level0_height = Counter(points[:,1]).most_common(1)[0][0]

        # import pdb; pdb.set_trace()
        # counts, values = np.histogram(points[:, 1])
        # if random_goal:
            # most frequenetly sampled height
            # self.level0_height = Counter(points[:,1]).most_common(1)[0][0]
        # else:
            # numLevels = len(self.env.sim._sim.semantic_scene.levels)
            # level_indices = np.argsort(counts)[-numLevels:]
            # level_values = np.sort(values[level_indices])
            # self.level0_height = level_values[0]

    def sample_start_state(self,fixed_floor=None):
        if fixed_floor is not None:
            point = self.env.sim.sample_navigable_point()
            while point[1] != self.floor_heights[fixed_floor]:
                point = self.env.sim.sample_navigable_point()
        else:
            point = self.env.sim.sample_navigable_point()

        rotation = hutil.quat_from_angle_axis(
            np.random.uniform(0, 2.0 * np.pi), np.array([0, 1, 0]))
        return point, rotation

    def distance_to_goal(self):
        return self._dist_to_goal(self.agent_state()[0])

    def _dist_to_goal(self,point):
        return min([self.env.sim.geodesic_distance(point, g) for g in self.goals])


    # Returns observation only
    def reset(self,fixed_floor=None,reachable=True):
        self.steps = 0
        self.env._sim.reset()
        if self.random_goal:
            def retry():
                pos, rot = self.sample_start_state(fixed_floor)
                self.env.sim.set_agent_state(pos, rot, 0)
                self.goals = [self.sample_start_state(fixed_floor)[0]]
        else:
            def retry():
                pos, rot = self.sample_start_state(fixed_floor)
                self.env.sim.set_agent_state(pos, rot, 0)
        retry()
        while self.distance_to_goal() == float('inf') and reachable:
            retry()
        return self.get_observation()

    def sample_reachable_goal(self,fixed_floor=None):
        g, _ = self.sample_start_state(fixed_floor)
        while self.env.sim.geodesic_distance(self.pos, g) == float('inf'):
            g, _ = self.sample_start_state(fixed_floor)
        return g

    def agent_state(self):
        agent_state = self.env.sim.get_agent_state()
        return agent_state.position, agent_state.rotation

    def set_agent_state(self,pos,ang):
        self.env.sim.set_agent_state(pos, ang, 0)

    def set_agent_position(self,pos):
        ang = self.agent_state()[1]
        self.env.sim.set_agent_state(pos, ang, 0)

    def set_agent_rotation(self,rot):
        pos = self.agent_state()[0]
        self.env.sim.set_agent_state(pos, rot, 0)

    @property
    def pos(self):
        return self.env.sim.get_agent_state().position

    @property
    def rot(self):
        return self.env.sim.get_agent_state().rotation

    @property
    def angle(self):
        agent_rot = hutil.quat_to_angle_axis(self.rot)
        # sometimes? reports 2pi rotation around x axis if no rotation
        if agent_rot[1][1] == 0:
            return 0
        else:
            return (agent_rot[0] * agent_rot[1][1])%(2*np.pi)

    # super hacky for now
    def get_all_obs(self):
        raw_obs = self.env.sim._sim.get_sensor_observations()
        obs =  self.env.sim.sensor_suite.get_observations(raw_obs)
        if 'rgb' in obs:
            obs['rgb'] = self.transform_obs(obs['rgb'][:, :, 0:3]) 
        return obs


    def get_observation(self,force_panorama=False):
        if (self.panorama or force_panorama):
            pos, rot = self.agent_state()
            angles = [
                rot * hutil.quat_from_angle_axis(rotation, np.array([0, 1, 0]))
                for rotation in np.arange(0, 1, 0.25) * 2 * np.pi
            ]

            def lam(ang):
                self.env.sim.set_agent_state(pos, ang, 0)
                return self.get_all_obs()

            images = [lam(ang) for ang in angles]
            self.env.sim.set_agent_state(pos, rot, 0)
            out = {}
            for k in images[0].keys():
                ims = [im[k] for im in images]
                if self.torchmode or self.torch_notrans:
                    out[k] =  torch.stack(ims)
                else:
                    out[k] =  np.stack(ims)
        else:
            out = self.get_all_obs()

        if self.crop_to_square:
            for k,o in out.items():
                if o.__class__ != np.ndarray: raise Exception(f'bad type')
                height,width = o.shape[-3:-1]
                start = int(width/2-height/2)
                # crop
                out[k] = o[...,:,start:start+height,:]
        return out

    # returns o,r,d,info
    # observation,reward,done?,info
    def step(self, action):
        self.steps = self.steps + 1
        # disallow 0 action which ends episode
        # shifts into the env action space with is 1 greater than this module
        pos,ang = self.agent_state()
        obs = self.env.sim.step(action + 1)
        # height change on stairs?
        # set to match the maxclimb value of the navmeshes
        height_deviations = [abs(self.pos[1] - e) > 0.2 for e in self.floor_heights ]
    
        if all(height_deviations) and not self.allow_stairs:
            print('height change detected')
            print('height: ',self.pos[1], 'floors:', self.floor_heights)
            self.set_agent_state(pos,ang)
            print('now: ',self.pos[1])
            # obs = self.get_observation()



        # could call this every time, caching the result above saves time
        # if self.panorama:
            # obs = self.get_observation()
        # elif 'rgb' in obs:
            # obs['rgb'] = self.transform_obs(obs['rgb'][:, :, 0:3]) 
        return self.get_observation(), 0, self.distance_to_goal() <= 2, None

    def transform_obs(self, obs):
        # imagenet transform
        if self.torchmode:
            return util.torch.to_imgnet(obs)
            # x = obs.float() / 255
            # x = x.permute(2, 0, 1)
            # x = x - (torch.tensor([0.485, 0.456, 0.406]).to(obs.device).view(-1, 1, 1).float())
            # x = x / (torch.tensor([0.229, 0.224, 0.225]).to(obs.device).view(-1, 1, 1).float())
            # return x
            # to check that the ouput matches the torch normalization
            # transformation = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],
            # [0.229, 0.224, 0.225])
            # ])
            # tens = transformation(obs).float()
            # tens
        return obs

    def close(self):
        return self.env.close()


if __name__ == '__main__':
    env = HabitatTestEnv( 'GIBSON_LOCATION/Ackermanville.glb', random_goal=True, panorama=False,torch_notrans=True,config_path='configs/tasks/pointnav_wide.yaml')
    env.get_observation()
    import pdb; pdb.set_trace()
    env.reset()
