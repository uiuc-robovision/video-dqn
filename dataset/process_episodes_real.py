import os
import numpy as np
from PIL import Image
import torch
from util import split_columns
import torch.optim as optim
from tqdm import tqdm
import util.torch
import argparse
import archs.inverse_action2 as inverse_action
import re
import pandas as pd
from torch.utils import data
from dataloaders.image_streams import ImageStream
import util.pd

detection_thresholds = [
    0.9700177907943726, 0.9738382697105408, 0.9512060284614563,
    0.7334915995597839, 0.7058018445968628
]

# branching dataset goals
transform = util.torch.imageNetTransformPIL()

score = lambda x: 0 if x is None else x.max()
score_detections = np.vectorize(score)
score_vals = lambda x: score_detections(x[:, 1])


# forward only
def calculate_steps(rewards):
    target_locs = []

    # identify steps with at target location
    for img_idx in range(len(rewards)):
        if rewards[img_idx]:
            target_locs.append(img_idx)

    steps = []
    for img_idx in range(len(rewards)):
        possible = list(filter(lambda x: x >= img_idx, target_locs))
        if len(possible) > 0:
            steps.append(min(possible) - img_idx)
        else:
            steps.append(float('inf'))
    steps = np.array(steps)
    return steps


def calculate_steps_negative(rewards):
    target_locs = []

    # identify steps with at target location
    for img_idx in range(len(rewards)):
        if rewards[img_idx]:
            target_locs.append(img_idx)

    if len(target_locs) > 0:
        steps = []
        for img_idx in range(len(rewards)):
            _, loc, _ = util.argmin(target_locs, lambda x: abs(img_idx - x))
            steps.append(loc - img_idx)
    else:
        # infinity
        steps = np.ones((rewards.shape[0], )) * float('inf')
    steps = np.array(steps)
    return steps



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process episodes')
    parser.add_argument('-g',
                        '--gpu',
                        dest='gpu',
                        default='0',
                        help='which gpu to run on')
    inverse_actions = True
    parser.add_argument('--location', default='dataset', help='folder root of what to process')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if inverse_actions:
        if not os.path.exists('inverse_model.torch'):
            import urllib.request
            url = 'http://matthewchang.web.illinois.edu/data/inverse_model.torch'
            print("\n\n\nDownloading Inverse Model File...")
            urllib.request.urlretrieve(url,'inverse_model.torch')
        # 40k version
        model_path = 'inverse_model.torch'
        # set up model
        model = inverse_action.model()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        model.cuda()

    results = np.load(f'{args.location}/frames/real_detections_raw.npy', allow_pickle=True)[()]

    skip = []
    all_samples = pd.DataFrame()
    for ep_id, res in tqdm(results.items()):
        if ep_id in skip:
            continue
        filters = np.load(f'{args.location}/filter_out/{ep_id}_filters.npy',
                          allow_pickle=True)[()]
        valid_frame = lambda x: x in filters[
            'indoor_locs'] and x not in filters['person_locs']
        im_ids = sorted(res.keys())
        if len(im_ids) == 0:
            continue

        filename = lambda x: f'dataset/frames/{ep_id}/%04d.jpg' % (x)
        valid_frame_new = lambda x: os.path.exists(filename(x))

        '''
        condition for valid_frame used below, i.e. valid_frame(i) and valid_frame_new(i), is actually redundant.
        valid_frame_new(i) should suffice since if the file exists at the location, it means it satisfied both the SLAM and the indoor/person criterion.
        Whereas valid_frame(i) simply checks for the latter.
        detections were ran on /videos_intersect which satisfy both conditions, so im_ids can't be a subset of frames valid according to both criterion.
        '''

        # works like python range
        episode_ranges = []
        episode_started = None
        for i in range(1, max(im_ids) + 2):
            if valid_frame(i) and valid_frame_new(i) and episode_started is None:
                episode_started = i
            elif episode_started is not None and not (valid_frame(i) and valid_frame_new(i)):
                episode_ranges.append((episode_started, i))
                episode_started = None
        if episode_started is not None:
            raise Exception(f'bad start')
        for start, stop in episode_ranges:
            if stop <= start + 3:
                continue
            samples = []
            ds = []
            for i in range(start, stop-3):
                samples.append(
                    (filename(i), filename(i + 3),ep_id,start,stop))
                ds.append(score_vals(res[i + 3]))

            ds = np.stack(ds)
            sample_frame = pd.DataFrame(
                samples,
                columns=['before_image', 'after_image','ep_id','im_start','im_stop'])
            util.pd.multi_add(sample_frame,ds,'detector_score')
            sparse_reward = (ds > detection_thresholds).astype(int)
            util.pd.multi_add(sample_frame,sparse_reward,'sparse_reward')

            steps = []
            for c in range(sparse_reward.shape[1]):
                steps.append(calculate_steps(sparse_reward[:, c]))
            steps = np.stack(steps, axis=1)
            util.pd.multi_add(sample_frame,steps,'steps_to_reward')

            steps = []
            for c in range(sparse_reward.shape[1]):
                steps.append(calculate_steps_negative(sparse_reward[:, c]))
            steps = np.stack(steps, axis=1)
            util.pd.multi_add(sample_frame,steps,'steps_to_reward_neg')
            all_samples = pd.concat((all_samples, sample_frame))

    if inverse_actions:
        print("Start inverse labeling")
        all_acts = []

        ims = np.stack(
            (all_samples['before_image'], all_samples['after_image']), axis=1)

        image_loader = data.DataLoader(ImageStream(ims),
                                       num_workers=4,
                                       batch_size=8)
        for be, ae in tqdm(image_loader):
            acts = model(be.cuda(), ae.cuda())[1]
            acts = acts.argmax(dim=1, keepdim=True).cpu().detach()
            all_acts.append(acts)

        all_samples['inverse_actions'] = torch.cat(all_acts).numpy()
    all_samples.reset_index(drop=True,inplace=True)
    all_samples.to_feather(f'{args.location}/data.feather')

