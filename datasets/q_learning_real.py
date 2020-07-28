import torch
from torch.utils import data
from PIL import Image
from util.torch import imageNetTransformPIL
import csv
import numpy as np
import os
import re
import util
import cv2
import pandas as pd
import util.pd
from functional import seq

# Confidence thresholds
detection_thresholds = [
    0.9700177907943726, 0.9738382697105408, 0.9512060284614563,
    0.7334915995597839, 0.7058018445968628
]

transform = imageNetTransformPIL()

score = lambda x: 0 if x is None else x.max()
score_detections = np.vectorize(score)
score_vals = lambda x: score_detections(x[:, 1])


class QLearningRealDataset(data.Dataset):
    def __init__(self,
                 location='/scratch/mc48/real_videos/filter_out/data.feather',
                 one_action=False,
                 value_learning=False,
                 inverse_actions=False,
                 previous_images=False,
                 confidence_reward=False,
                 slam_actions=False,
                 gamma=0.99):
        self.samples = pd.read_feather(location)
        self.value_learning = value_learning
        self.confidence_reward = confidence_reward
        self.slam_actions = slam_actions
        self.one_action = one_action
        self.inverse_actions = inverse_actions
        self.gamma = gamma
        self.previous_images = previous_images

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def reward_percentage(self):
        rewards = util.pd.multi_get(self.samples, 'sparse_reward')
        return (rewards.max(axis=1) > 0).sum() / rewards.shape[0]

    # returns before image, after image, action, reward, terminal
    def __getitem__(self, index):
        sample = self.samples.loc[index]
        if self.previous_images:
            start = sample['im_start']

            def get_ims(path):
                matches = re.match('(.*?/)(\d+).jpg', path)
                prefix, im_id = matches[1], int(matches[2])
                ids = [max(im_id - i, start) for i in range(4)]
                return torch.stack([
                    transform(Image.open(prefix + "%04d.jpg" % (i)))
                    for i in ids
                ])

            bi = get_ims(sample['before_image'])
            ai = get_ims(sample['after_image'])
        else:
            bi = transform(Image.open(sample['before_image']))
            ai = transform(Image.open(sample['after_image']))

        detections = util.pd.multi_get(sample, 'detector_score')
        steps_to_reward = util.pd.multi_get(sample, 'steps_to_reward')

        if self.confidence_reward:
            reward = detections
            termainl = np.zeros_like(reward)
        else:
            reward = (detections > detection_thresholds).astype(np.int)
            terminal = reward
        valid_mask = np.ones_like(reward)
        gt = np.nan
        if self.value_learning:
            gt = np.power(np.ones((5, )) * self.gamma, steps_to_reward)
            # don't process samples with no reward
            gt[steps_to_reward == np.inf] = np.nan
        if self.inverse_actions:
            action = sample['inverse_actions']
        elif self.slam_actions:
            raise 'not implemented'
        elif self.one_action:
            action = 0
        else:
            raise Exception(f'not implemented')
        return bi, ai, action, reward, reward, gt, valid_mask


if __name__ == '__main__':
    dataset = QLearningRealDataset(inverse_actions=True)
    import pdb; pdb.set_trace()
    dataset[0]
    
    samples = pd.read_feather(f'/scratch/mc48/real_videos/frames/data.feather')
    # samples = pd.read_feather(f'/scratch/mc48/real_videos/frames/data.feather')
    samples['steps_to_reward1']
    steps_to_reward = util.pd.multi_get(samples, 'steps_to_reward')
    steps_to_reward
    print(dataset[0])
    import pdb
    print(len(dataset))
    pdb.set_trace()
    # dataset[0][2].shape
    # import pdb; pdb.set_trace()
