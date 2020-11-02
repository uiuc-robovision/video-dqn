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

transform = imageNetTransformPIL()

class DetectorRealDataset(data.Dataset):
    def __init__(self,files,predictor):
        self.samples = files
        # to get proper transformation
        self.predictor = predictor

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    # returns before image, after image, action, reward, terminal
    def __getitem__(self, index):
        info,path = self.samples[index]
        im = cv2.imread(path)
        if self.predictor.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            im = im[:, :, ::-1]
        image = self.predictor.transform_gen.get_transform(im).apply_image(im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        return info, image
