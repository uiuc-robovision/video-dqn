# coredump without this for some reason
import torch
import os
import numpy as np

import cv2
from matplotlib import pyplot as plt
import re
import util.cv2
import util
from util.plt import show
from time import sleep, time
import json
import random
from async_data_writer import AsyncLambdaRunner, ensure_folders, numpy_writer
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from datasets.detector_real import DetectorRealDataset
from torch.utils import data

os.environ["CUDA_VISIBLE_DEVICES"]='4'

vid_location = 'dataset/frames'
class_labels = sorted(['bed', 'chair', 'couch', 'dining table', 'toilet'])

# threshes = [32079.0, 11088.0, 24084.0, 13872.0, 5343.0]

def get_predictor():
    cfg = get_cfg()
    cfg.merge_from_file('configs/detectron/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    return DefaultPredictor(cfg)

predictor = get_predictor()
class_indices = [predictor.metadata.thing_classes.index(c) for c in class_labels]

def pred_to_score(prediction):
    def lam(ci):
        pr = prediction['instances']
        box_mask = pr.pred_classes == ci
        boxes = pr.pred_boxes.tensor[box_mask].detach().cpu().numpy()
        scores = pr.scores[box_mask].detach().cpu().numpy()
        if len(boxes) == 0: boxes= None
        if len(scores) == 0: scores= None
        return (boxes,scores)

    res = list(map(lam,class_indices))
    arr = np.empty((5,2),dtype='O') 
    arr[:] = res
    return arr

episodes = [ f.name for f in os.scandir(vid_location) if f.is_dir() ]

results = {}

for ep in tqdm(episodes):
    folder = f'{vid_location}/{ep}'
    fils = util.files(folder,'(\d+).jpg')
    inds = [int(re.match('(\d+).jpg',fil)[1]) for fil in fils]
    full_files = [folder+'/'+f for f in fils]
    res = list(zip(inds,full_files))
    dataset = DetectorRealDataset(res,predictor=predictor)
    loader = data.DataLoader(dataset,batch_size = 4,num_workers=4)
    generator = iter(loader)
    ep_res = {}
    for batch in generator:
        ins = [{'image': im} for im in batch[1]]
        res = predictor.model(ins)
        vals = list(map(pred_to_score,res))
        ind = batch[0]
        for i in range(len(ind)):
            ep_res[ind[i].item()] = vals[i]
    results[ep]=ep_res
np.save(f'{vid_location}/real_detections_raw',results)
