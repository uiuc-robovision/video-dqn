from PIL import Image
import torch
import os
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from habitat.utils.visualizations import maps
from habitat_sim import utils as hutil
import util
import util.torch
import util.habitat
from util.plt import *
import cv2
import re
from datasets.habitat_visualization_data_gibson import HabitatQVisualizationDatasetGibson
from torch.utils import data
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

MAP_RESOLUTION = 1000
data_folder = f'./rendered_data/val_data-{MAP_RESOLUTION}'


# def build_map(config,model_number):
    # if config.PANORAMA:
        # modelFunc = HabitatDQNModelPanorama
    # else:
        # modelFunc = HabitatDQN

    # model_loc = f'{config.folder}/models/sample{model_number}.torch'
    # snapshot = torch.load(model_loc)
    # print(f'Loading model from: {model_loc}')
    # sigmoid = config.LOSS_CLIP == 'sigmoid'
    # model = modelFunc(3,sigmoid=sigmoid)
    # model.load_state_dict(snapshot['model_state_dict'])
    # model.eval()
    # model.to(config.device)


    # maps = []
    # for ori in range(0,4):
        # new_map = np.zeros((MAP_RESOLUTION, MAP_RESOLUTION))
        # # free_map = np.zeros((MAP_RESOLUTION, MAP_RESOLUTION))
        # dataset = HabitatQVisualizationDataset(f'rendered_data/val_data-{MAP_RESOLUTION}',panorama=config.PANORAMA,orientation=ori)

        # params = {'batch_size': 32, 'num_workers': 6}
        # data_generator = data.DataLoader(dataset, **params)
        # points = 0

        # for batch in tqdm(data_generator):
            # row,col,images = batch
            # values = model(images.to(config.device)).max(1).values.detach().cpu()
            # new_map[row,col]=values
            # # free_map[row,col]=1
            # points += len(row)
        # maps.append(new_map)
    # # np.save("free_space_map",free_map)
    # return maps

def build_map_gibson(config,model,house,floor):
    import gibson_info
    model.to(config.device)
    model.eval()
    resolution = 1500
    if config.DATASET == 'real':
        data_root = f'rendered_data/vis_data_{resolution}_real/{house.name}_{floor}'
    else:
        data_root = f'rendered_data/vis_data_{resolution}/{house.name}_{floor}'
    info = np.load(f'{data_root}/info.npy',allow_pickle=True)[()]
    # resolution= info['map_resolution']
    agent_location = info['agent_location']

    maps = []
    variance_maps = []
    free_map = np.zeros((resolution, resolution))
    for ori in range(0,4):
        new_map = np.zeros((resolution, resolution,len(gibson_info.class_labels)))
        var_map = np.zeros((resolution, resolution,len(gibson_info.class_labels)))
        dataset = HabitatQVisualizationDatasetGibson(data_root,panorama=config.PANORAMA or config.PREVIOUS_IMAGES,orientation=ori)

        params = {'batch_size': 32, 'num_workers': 6}
        data_generator = data.DataLoader(dataset, **params)

        for batch in tqdm(data_generator):
            row,col,images = batch
            if config.DISTRIBUTIONAL:
                outputs = model(images.to(config.device))
                means = outputs[...,0].max(2).values.detach().cpu()
                var = outputs[...,1].max(2).values.detach().cpu()
                new_map[row,col]=means
                var_map[row,col]=var
                free_map[row,col]=1
            else:
                values = model(images.to(config.device)).max(2).values.detach().cpu()
                new_map[row,col]=values
                free_map[row,col]=1
        maps.append(new_map)
        variance_maps.append(var_map)

    # aggregate map and draw colors appropriately
    agg_map_uncropped = np.stack(maps).max(0)
    var_agg_map_uncropped = np.stack(variance_maps).max(0)
    # import pdb; pdb.set_trace()
    figs = []
    types = ['mean', 'variance']
    for ty in types:
        for direct in [0,1,2,3,'max']:
            for i,label in enumerate(gibson_info.class_labels):
                locs = gibson_info.relevant_locations(agent_location,house.object_locations_for_habitat_dest[label])
                mark_locations = [util.habitat.to_grid(l,resolution) for l in locs]

                map_base = maps
                if ty == 'variance':
                    map_base = variance_maps

                if direct == 'max':
                    cur_map = agg_map_uncropped[:,:,i]
                    if ty == 'variance':
                        cur_map = var_agg_map_uncropped[:,:,i]
                else:
                    cur_map = map_base[direct][:,:,i]

                values = cur_map[free_map==1]
                # vmin = 0
                # vmax = 1
                vmin = values.min()
                vmax = values.max() 
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                final_map = np.zeros((resolution, resolution, 3))
                cmap=matplotlib.cm.get_cmap('viridis')
                final_map[free_map==1,:] = cmap(norm(values))[:,:3]

                for row,col in mark_locations:
                    final_map[row,col,:] = [1,0,0]

                row_crop,col_crop = util.habitat.crop_range(free_map)
                cropped_final = final_map[row_crop[0]:row_crop[1],col_crop[0]:col_crop[1],:]
                fig = plt.Figure()
                ax = fig.subplots()
                ax.set_title(f'{label}, {direct}, {ty}')
                pos = ax.imshow(cropped_final,cmap='viridis',vmin=vmin,vmax=vmax)
                fig.colorbar(pos,ax=ax)
                figs.append(fig)
    return figs


if __name__ == '__main__':
    from experiment_config import ExperimentConfig
    config = ExperimentConfig(f"logs/trained_ground_truth_panorama", device='cuda:2')
    build_map(config,99)
