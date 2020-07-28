#! /scratch/mc48/venv/bin/python
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import torchvision.models as models
import numpy as np
import torch
import torch.optim as optim
from archs.HabitatDQNMultiAction import HabitatDQNMultiAction
# from datasets.gibson import GibsonDataset
from datasets.q_learning_real import QLearningRealDataset
from torch.utils import data
import util
import os
from gibson_info import get_houses, class_labels, get_house

from visualize_value import build_map_gibson
from matplotlib import pyplot as plt
import torchvision.utils

GIBSON_DATASETS = [
    'gibson', 'gibson_medium_40k', 'gibson_medium', 'gibson_medium_inverse',
    'gibson_medium_noisy_40k', 'gibson_medium_noisy',
    'gibson_medium_noisy_value', 'gibson_medium_noisy_40k_value',
    'gibson_branch', 'multi_target', 'multi_target_longer', 'explore',
    'branch_updated'
]

# fix for errors loading data on az005
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024, rlimit[1]))


def build_model(config):
    sigmoid = config.LOSS_CLIP == 'sigmoid'
    if config.VALUE_LEARNING or config.ONE_ACTION:
        actions = 1
    else:
        actions = 3
    model = HabitatDQNMultiAction(
        actions,
        5,
        extra_capacity=(config.ARCHITECTURE == 'extra_capacity'),
        panorama=(config.PANORAMA or config.PREVIOUS_IMAGES))
    return model.to(config.device)


def load_model_number(config, number):
    model = build_model(config)
    model_loc = f'{config.folder}/models/sample{number}.torch'
    snapshot = torch.load(model_loc, map_location=config.device)
    print(f'Loading model from: {model_loc}')
    model.load_state_dict(snapshot['model_state_dict'])
    return model


def loopLoader(loader):
    i = iter(loader)
    while True:
        try:
            yield next(i)
        except StopIteration:
            print("reset iterator")
            i = iter(loader)

def visualize_house(config, model, house, floor, sample_number):
    print(f'render sample: {sample_number} on {house.name}{floor}')
    figs = build_map_gibson(config, model, house, floor)
    images = np.array([util.plt.fig2img(x) for x in figs])[..., :-1]
    # make grid and convert to torch format
    grid = torchvision.utils.make_grid(torch.tensor(images).permute(
        0, 3, 1, 2),
                                       nrow=5)
    config.writer.add_image(
        f"value_map_{house.name}{floor}({house.data['split_tiny']})",
        grid,
        global_step=sample_number)


def run_train(config, resume_from=-1):
    torch.set_num_threads(1)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Using: {config.device}")
    # gibson_houses = get_houses()
    houses_to_render = [("Allensville", 0), ("Beechwood", 1), ("Darden", 0),
                        ("Darden", 1), ("Markleeville", 1), ("Merom", 1),
                        ("Arkansaw", 1), ("Pomaria", 0)]
    if config.DATASET == 'branch_updated':
        houses_to_render = [("Arkansaw", 1)]
    # def house_by_name(name):
    # return [g for g in gibson_houses if g.name == name][0]
    houses_to_render = list(
        map(lambda x: (get_house(x[0]), x[1]), houses_to_render))

    params = {'batch_size': 16, 'num_workers': 8, 'drop_last': True}
    #using different evaluation data
    if config.DATASET == 'nobias':
        datafile = 'rendered_data/randomized_data_nobias_S9hNv5qa7GM/data.npy'
        dataset = HabitatQRandomizedDataset(datafile,
                                            gamma=config.GAMMA,
                                            panorama=config.PANORAMA,
                                            known=config.TRAIN_ON_GROUND_TRUTH)
    elif config.DATASET == 'real':
        datafile = '/scratch/mc48/real_videos/filter_out/data.feather'
        dataset = QLearningRealDataset(
            datafile,
            one_action=True,
            confidence_reward=config.CONFIDENCE_REWARD,
            value_learning=config.VALUE_LEARNING,
            inverse_actions=config.USE_INVERSE_ACTIONS,
            previous_images=config.PREVIOUS_IMAGES)
        known_gt_data = QLearningRealDataset(datafile, one_action=True)
    elif config.DATASET in GIBSON_DATASETS:
        reward_dist = 1
        if config.DATASET == "gibson":
            datafile = 'rendered_data/gibson_30_fixed/train_data.npy'
        elif config.DATASET == 'gibson_medium':
            datafile = 'rendered_data/gibson_medium/train_data.npy'
        elif config.DATASET == 'gibson_medium_40k':
            datafile = 'rendered_data/gibson_medium/data_40k.npy'
        elif config.DATASET == 'gibson_medium_noisy':
            datafile = '/ssd/mc48/habitat_data/gibson_medium_qlearning_train_e0.2/medium_qlearning_train_data.npy'
        elif config.DATASET == 'gibson_medium_noisy_40k':
            datafile = '/ssd/mc48/habitat_data/gibson_medium_qlearning_train_e0.2/data_40k.npy'
            # datafile = '/ssd/mc48/habitat_data/gibson_medium_qlearning_train_e0.2/data_40k_new_gt.npy'
        elif config.DATASET == 'gibson_branch':
            datafile = '/ssd/mc48/habitat_data/gibson_branching_e0.2/data.npy'
        elif config.DATASET == 'branch_updated':
            if config.VALUE_LEARNING:
                datafile = '/ssd/mc48/habitat_data/gibson_branching_updated_e0.2/data_direct_value_geodesic.npy'
            else:
                datafile = '/ssd/mc48/habitat_data/gibson_branching_updated_e0.2/data_40k.npy'
        elif config.DATASET == 'gibson_medium_noisy_value':
            datafile = '/scratch/arjung2/data/gibson_medium_qlearning_e0.2_train_data.npy'
        elif config.DATASET == 'gibson_medium_noisy_40k_value':
            # datafile = '/scratch/arjung2/data/gibson_medium_qlearning_e0.2_train_data.npy'
            if not config.VALUE_LEARNING:
                raise Exception('dataset doesnt exist')
            if config.DETECTION_REWARDS:
                if config.PANORAMA:
                    datafile = '/ssd/mc48/habitat_data/gibson_medium_qlearning_train_e0.2/direct_value_data_detection_pano.npy'
                else:
                    datafile = '/ssd/mc48/habitat_data/gibson_medium_qlearning_train_e0.2/direct_value_nopano_data.npy'
            else:
                datafile = '/ssd/mc48/habitat_data/gibson_medium_qlearning_train_e0.2/direct_value_data_geodesic.npy'
        elif config.DATASET == 'multi_target':
            if config.VALUE_LEARNING:
                if config.PANORAMA:
                    raise Exception(f'Not generated')
                else:
                    datafile = '/scratch/mc48/habitat_data/gibson_medium_qlearning_train_e0.2_multi_target/direct_value_data_detection.npy'
            else:
                datafile = '/scratch/mc48/habitat_data/gibson_medium_qlearning_train_e0.2_multi_target/data_40k.npy'
        elif config.DATASET == 'multi_target_longer':
            if config.VALUE_LEARNING:
                if config.PANORAMA:
                    raise Exception(f'Not generated')
                else:
                    datafile = '/scratch/mc48/habitat_data/gibson_medium_qlearning_train_e0.2_longer/direct_value_data_detection.npy'
            else:
                datafile = '/scratch/mc48/habitat_data/gibson_medium_qlearning_train_e0.2_longer/data_40k.npy'
        elif config.DATASET == 'explore':
            root = '/scratch/mc48/habitat_data/gibson_medium_qlearning_train_e0.2_explore/'
            if config.VALUE_LEARNING:
                if config.PANORAMA:
                    datafile = root + '/direct_value_data_detection_pano.npy'
                else:
                    datafile = root + '/direct_value_data_detection.npy'
            else:
                datafile = root + 'data_40k.npy'
        dataset = GibsonDataset(datafile,
                                panorama=config.PANORAMA,
                                class_label=config.CLASS_LABEL,
                                inverse_action=config.USE_INVERSE_ACTIONS,
                                random_actions=config.RANDOM_ACTIONS,
                                value_learning=config.VALUE_LEARNING,
                                reward_dist=reward_dist,
                                one_action=config.ONE_ACTION,
                                prev_images=config.PREVIOUS_IMAGES)
        known_gt_data = GibsonDataset(
            datafile,
            panorama=config.PANORAMA,
            known=True,
            class_label=config.CLASS_LABEL,
            inverse_action=config.USE_INVERSE_ACTIONS,
            random_actions=config.RANDOM_ACTIONS,
            value_learning=config.VALUE_LEARNING,
            semantic_rewards=config.SEMANTIC_REWARDS,
            detector_rewards=config.DETECTION_REWARDS,
            reward_dist=reward_dist,
            one_action=config.ONE_ACTION,
            prev_images=config.PREVIOUS_IMAGES)
    elif config.DATASET == 'bias_single':
        datafile = 'rendered_data/randomized_data/data.npy'
        dataset = HabitatQRandomizedDataset(datafile,
                                            gamma=config.GAMMA,
                                            panorama=config.PANORAMA,
                                            known=config.TRAIN_ON_GROUND_TRUTH)
    elif config.DATASET == 'reacher':
        datafile = 'reacher_data/custom.npy'
        dataset = ReacherDataset(datafile,
                                 gamma=config.GAMMA,
                                 tip_pos=False,
                                 has_reward=False)
        render_dataset = ReacherDataset(datafile,
                                        gamma=config.GAMMA,
                                        tip_pos=True,
                                        downsample=int(5e4))

    elif config.DATASET == 'reacher_replay':
        datafile = 'reacher_data/replay.npy'
        dataset = ReacherDataset(datafile,
                                 gamma=config.GAMMA,
                                 tip_pos=False,
                                 has_reward=True)
        render_dataset = ReacherDataset(datafile,
                                        gamma=config.GAMMA,
                                        tip_pos=True,
                                        downsample=int(5e4),
                                        has_reward=True)

    elif config.DATASET == 'reacher_replay_override':
        datafile = 'reacher_data/replay.npy'
        dataset = ReacherDataset(datafile,
                                 gamma=config.GAMMA,
                                 tip_pos=False,
                                 has_reward=True,
                                 override_reward=True)
        render_dataset = ReacherDataset(datafile,
                                        gamma=config.GAMMA,
                                        tip_pos=True,
                                        downsample=int(5e4),
                                        has_reward=True,
                                        override_reward=True)

    print(f'Load data from {datafile}')
    print(f'Reward Ration: {dataset.reward_percentage()}')

    test_size = int(params['batch_size'] * 10)
    # train_data, test_data = data.random_split( dataset, [len(dataset) - test_size, test_size])
    train_data = dataset
    training_generator = data.DataLoader(train_data, **params, shuffle=True)
    # threshold at roughly 5 meters

    _, test_data = data.random_split(
        known_gt_data, [len(known_gt_data) - test_size, test_size])

    # training_generator = data.DataLoader(train_data, **params,sampler=WeightedRandomSampler(train_data.weights,len(train_data),replacement=True))
    eval_generator = data.DataLoader(test_data, **params, shuffle=True)
    print(len(train_data))
    print(len(test_data))

    # Training

    model = build_model(config)
    target_net = build_model(config)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    def process_batch(batch, compare_ground_truth=False, batch_number=None):
        before, after, act, rew, term, ground_truth, valid_mask = [
            x.to(config.device) for x in batch
        ]
        # batch x 5 x 3
        before_values = model(before)
        # get the current predictions for q values of the actions taken
        # -1 x 5
        action_indices = act.view(-1, 1).repeat(1, 5)

        # -1 x 5
        Q_b = before_values.gather(2, action_indices.unsqueeze(2)).squeeze()

        if not compare_ground_truth:
            after_values = target_net(after)
            # use his for double dqn
            model_after_values = model(after)
            # use this for regular dqn
            # model_after_values = after_values

            #select the best action acording to the online model
            # -1 x 5
            best_actions = model_after_values.argmax(-1)
            # best_actions = model_after_values.argmax(1,-1).view(-1, 1)

            # get q values of best action from next state
            # detach so no gradients flow back
            after_values.shape
            # -1 x 5
            Q_a = after_values.gather(
                2, best_actions.unsqueeze(2)).detach().squeeze()

            # remove after value for terminal states, so
            # only the reward is fitted
            Q_a = Q_a * (1 - term.float())
            if config.LINEAR:
                learn_targets = rew.float() + (Q_a - 0.1)
            else:
                learn_targets = rew.float() + config.GAMMA * Q_a
            if config.LOSS_CLIP == 'rect':
                learn_targets = torch.clamp(learn_targets, max=1, min=0)
            losses = (0.5 * (Q_b - learn_targets)**2)
            if config.REMOVE_BEFORE_REWARD:
                losses = losses * valid_mask
        else:
            # for ground truth values
            if config.VALUE_LEARNING:
                mask = (1 - torch.isnan(ground_truth).int())
                gt = ground_truth.clone()
                gt[torch.isnan(ground_truth)] = 0
                losses = 0.5 * (Q_b * mask - gt.float())**2
            else:
                losses = 0.5 * (Q_b - ground_truth.float())**2

        loss = losses.mean()
        return loss

    metrics = {
        'losses': [],
        'eval_losses': [],
    }
    iterator = loopLoader(training_generator)

    os.system(f'mkdir {config.folder}/models')
    sample_number = resume_from + 1

    if resume_from > -1:
        # model_loc = f'{config.folder}/models/epoch{resume_from}.torch'
        model_loc = f'{config.folder}/models/sample{resume_from}.torch'
        snapshot = torch.load(model_loc, map_location=config.device)
        print(f'Loading model from: {model_loc}')
        model.load_state_dict(snapshot['model_state_dict'])
        optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    # test loading gt net
    if config.BOOTSTRAP:
        print('\n\nBOOTSTRAP\n\n')
        model_loc = f'logs/trained_gt_0.99/models/epoch99.torch'
        snapshot = torch.load(model_loc, map_location=config.device)
        print(f'Loading model from: {model_loc}')
        model.load_state_dict(snapshot['model_state_dict'])
        optimizer.load_state_dict(snapshot['optimizer_state_dict'])

    target_net.load_state_dict(model.state_dict())
    running_loss = None
    # for epoch in range(resume_from+1,100):
    while sample_number < config.NUM_STEPS:
        # update target network if needed
        sample_number += 1
        # visualize_house(config,model,"Beechwood",sample_number)
        if sample_number % config.TARGET_UPDATE_INTERVAL == 0:
            target_net.load_state_dict(model.state_dict())

        # perform update
        # model.train()
        # set training mode
        model.set_train()
        optimizer.zero_grad()
        loss = process_batch(next(iterator),
                             compare_ground_truth=config.TRAIN_ON_GROUND_TRUTH,
                             batch_number=sample_number)
        loss.backward()
        optimizer.step()
        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * 0.99 + loss.item() * 0.01
        print(
            f'\rbatch:{sample_number}/{config.NUM_STEPS} avg_loss: {running_loss}',
            end="")

        if sample_number % 100 == 0:
            config.writer.add_scalar('avg_q_loss/train', running_loss,
                                     sample_number)

        # checkpoint and eval
        if sample_number % config.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    'sample_number': sample_number,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f'{config.folder}/models/sample{sample_number}.torch')
            for house, floor in houses_to_render:
                visualize_house(config, model, house, floor, sample_number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train q network')
    parser.add_argument('-g',
                        '--gpu',
                        dest='gpu',
                        default='0',
                        help='which gpu to run on')

    parser.add_argument('-r',
                        '--resume',
                        dest='resume',
                        action="store_true",
                        help='resume from last epoch?')

    parser.add_argument('-d',
                        '--delete',
                        dest='delete',
                        action="store_true",
                        help='delete stored tensorboard data')

    parser.add_argument('config', help='folder containing config file')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from experiment_config import ExperimentConfig
    config = ExperimentConfig(args.config,
                              device='cuda',
                              remove=args.delete,
                              resume=args.resume)

    with open(f'{config.folder}/log', "w") as text_file:
        text_file.write(f"Running with config ({str(config.cfg)})")

    if args.resume:
        models = os.popen(f'ls {config.folder}/models').read().split()

        if len(models) == 0:
            run_train(config)
        else:
            latest_model = max([int(n[6:-6]) for n in models])
            print(f"Resuming from: {latest_model}")
            run_train(config, latest_model)
    else:
        run_train(config)
