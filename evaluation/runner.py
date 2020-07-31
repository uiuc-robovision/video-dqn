# limit numpy threads
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import torch
from torch.utils import data
import util
import util.habitat as habutil
from gibson_info import get_houses_medium, class_labels, get_houses, get_house, relevant_locations
from matplotlib import pyplot as plt
from habitat.utils.visualizations import maps
import habitat_sim.utils.common as hutil
from train_q_network import load_model_number
from tqdm import tqdm
from habitat_test_env import HabitatTestEnv
import random
import queue
import math
from evaluation.policy_defaults import name_from_config, load_file
from defaults import get_cfg_defaults
from evaluate import ours_evaluate 
from experiment_config import ExperimentConfig
from disk_logger import DiskLogger

def run_policy(config, args):
    print(config)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_folder = f'{config.RESULT_LOCATION}/{name_from_config(config)}'
    print('Log Folder:', log_folder)
    logger = DiskLogger(log_folder, checkpointTime=(60 * 30))

    if config.DATASET == 'val':
        episode_location = 'evaluation/val_episodes.npy'

    episodes = np.load(episode_location, allow_pickle=True)

    house_name = ""
    env = None

    model_config = ExperimentConfig(config.MODEL_CONFIG_LOCATION,
                                    tensorboard=False)
    model = load_model_number(model_config, config.MODEL_NUMBER)
    model.eval()

    for epind in tqdm(range(len(episodes))):
        # for epind in tqdm(hard_epinds):
        ep = episodes[epind]
        print(f"EP_INDEX: {epind}")
        print(ep)

        hn, floor, class_label, goal_dist, pos, rot = ep
        if house_name != hn:
            if env is not None:
                env.close()
            house_name = hn
            house = get_house(house_name)
            params = {
                'num_floors': house.num_floors,
                'allow_stairs': config.STAIRS,
                'panorama': config.SCORE == 'model'
                and config.MODEL_CONFIG.PANORAMA,
                'torchmode': False,
                'config_path': 'configs/tasks/pointnav_rgbd.yaml'
            }
            env = house.get_env(**params)

        loc = env.sample_start_state(int(floor))[0]
        # goals for the given floor
        goals = relevant_locations(
            loc, house.object_locations_for_habitat_dest[class_label])
        env.goals = goals

        env.set_agent_state(pos, rot)
        vis = args.visualize or epind % 100 == 0
        output = ours_evaluate(config, env, ep, house, epind, model, vis,
                               model_config)
        if not args.debug: logger.write(epind, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate policy')
    parser.add_argument('-g',
                        '--gpu',
                        dest='gpu',
                        default='0',
                        help='which gpu to run on')
    parser.add_argument('-p',
                        '--profile',
                        dest='profile',
                        action='store_true',
                        help='profile')
    parser.add_argument('-d',
                        '--debug',
                        dest='debug',
                        action='store_true',
                        help='debug mode, no writing to results files')

    parser.add_argument('-s',
                        '--start',
                        dest='start',
                        default='0',
                        type=int,
                        help='which sample to start on ')
    parser.add_argument('-r',
                        '--resume',
                        dest='resume',
                        action='store_true',
                        help='resume last uncompleted episode')
    parser.add_argument('--episodes',
                        dest='episodes_to_run',
                        help='episodes to run')
    parser.add_argument('-v',
                        '--visualize',
                        dest='visualize',
                        action='store_true',
                        help='visualize')
    parser.add_argument('config', help='folder containing config file')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_file(args.config)
    run_policy(config, args)
