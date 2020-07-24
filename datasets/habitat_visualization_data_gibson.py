import torch
from torch.utils import data
from PIL import Image
from util.torch import imageNetTransformPIL
import csv
import numpy as np
import os
import re

transform = imageNetTransformPIL()

class HabitatQVisualizationDatasetGibson(data.Dataset):
    def __init__(self,data_folder,orientation,panorama):
        self.samples = []
        self.data_folder = data_folder
        self.orientation=orientation
        self.panorama=panorama
        points = os.popen(f'ls {data_folder} | grep jpg').read().split()
        def lam(p):
            res = re.search("(\d+)-(\d+)-\d+.jpg",p)
            return (int(res[1]),int(res[2]))
        self.samples = list(set([lam(p) for p in points]))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    # returns before image, after image, action, reward, terminal
    def __getitem__(self, index):
        'Generates one sample of data'
        row, col = self.samples[index]
        images = torch.stack([transform(Image.open(f"{self.data_folder}/{row}-{col}-{i}.jpg"))  for i in range(0,4)])
        if self.panorama:
            rotated_images = torch.cat([images[self.orientation:, ...], images[:self.orientation, ...]])
        else:
            rotated_images = images[self.orientation]

        return row, col, rotated_images

if __name__ == '__main__':
    dataset = HabitatQVisualizationDatasetGibson('rendered_data/vis_data_scaled_15000/Allensville',panorama=True,orientation=0)
    # dataset[0][2].shape
    # import pdb; pdb.set_trace()

