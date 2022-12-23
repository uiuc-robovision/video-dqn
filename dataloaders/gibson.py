import torch
from torch.utils import data
from PIL import Image
from util import split_columns,sample_axis
import csv
import numpy as np
import torchvision.transforms as transforms
from os import path
import random
import matplotlib.pyplot as plt

def imageNetTransformPIL(size=224):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        # transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform = imageNetTransformPIL()
valid_labels = sorted(['bed', 'chair', 'couch', 'dining table', 'toilet'])

class GibsonDatasetPair(data.Dataset):
    def __init__(self,
                 file_location,
                 gamma=0.9,
                 panorama=False,
                 min_dist=0,
                 known=False,reward_dist=1,class_label='toilet'):
        self.gamma = gamma
        self.panorama = panorama
        self.reward_dist = reward_dist
        self.class_label = class_label
        self.classes = len(valid_labels) if class_label == 'all' else 1

        samples = np.load(file_location)
        self.samples = samples
        # select the column of min_dists specified by the class label
        if class_label != 'all':
            rest,min_dists = split_columns(samples,[17,len(valid_labels)])
            class_dists = min_dists.astype(np.float)[:,valid_labels.index(class_label)]
            self.samples = np.concatenate((rest,class_dists[:,None]),axis=1)

        if known and class_label != 'all':
            _,_,_, _,_,_,_,min_dists = split_columns(self.samples,[1,3,4,1,3,4,1,1])
            known_samples = min_dists.astype(np.float)[:,0] <= reward_dist
            self.samples=self.samples[known_samples,:]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    # returns before image, after image, action, reward, terminal
    def __getitem__(self, index):
        'Generates one sample of data'
        bel,before_pos,before_ang, ael,after_pos,after_ang,action,min_dists = split_columns(self.samples[index,:],[1,3,4,1,3,4,1,self.classes])
        bel=bel[0]
        ael=ael[0]

        if self.panorama:
            orientations = list(range(0, 4))
            be = torch.stack([
                transform(Image.open(f'{bel}/{o}.jpg')) for o in orientations
            ])
            ae = torch.stack([
                transform(Image.open(f'{ael}/{o}.jpg')) for o in orientations
            ])
        else:
            be = transform(Image.open(f'//scratch/mc48/habitat_test/{bel}/0.jpg'))
            ae = transform(Image.open(f'//scratch/mc48/habitat_test/{ael}/0.jpg'))
        min_dists = min_dists.astype(np.float)
        reward = (min_dists <= self.reward_dist).astype(np.int)
        gt = np.power(np.ones((self.classes))*self.gamma,min_dists)
        term = reward
        return be, ae, int(action)-1, reward, term, gt
