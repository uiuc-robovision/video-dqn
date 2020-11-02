import gibson_info
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as hutil
from util.torch import numpy_to_imgnet, get_device
import numpy as np
import cv2
import util.cv2
import torch
from tqdm import tqdm
import util.habitat as habutil
import matplotlib
from matplotlib import pyplot as plt
import os
from gibson_info import class_labels

# os.environ["CUDA_VISIBLE_DEVICES"]='4'

map_resolution = 1200
to_grid = lambda x: habutil.to_grid(x, map_resolution)


def min_dist(env, goals, point=None):
    if point is None:
        point = env.agent_state()[0]
    if len(goals) == 0:
        return float('inf')
    return min([env.env.sim.geodesic_distance(point, g) for g in goals])


def min_dists(env, goals, point=None):
    return np.array([min_dist(env, gs, point) for gs in goals])


def join_images(ims, values=None, br_text="", bl_text=''):
    cols = ims[0].shape[1]
    scale = (4.0 / len(ims)) - 0.05
    rng = int(scale * cols / 2)

    ims = list(reversed(ims))
    ims = np.array(
        [im[:, (cols // 2) - rng:(cols // 2) + rng, :] for im in ims])
    # black bar for last column
    ims[:, :, -1] = 0
    joined_ims = np.concatenate(ims, axis=1)
    if values is None:
        return joined_ims
    else:
        values = list(reversed(values))

    annotations = []
    for val in values:
        text = np.ones((50, rng * 2, 3)) * 255
        # write text if not finlizing
        cv2.putText(text, "{:.2f}".format(val), (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        annotations.append(text)

    text_row = np.concatenate(annotations, axis=1).astype(np.uint8)

    text_width = cv2.getTextSize(br_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                 1)[0][0]
    cv2.putText(text_row, br_text, (text_row.shape[1] - text_width - 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(text_row, bl_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    combined = np.concatenate((joined_ims, text_row), axis=0)
    return combined


def vis_panorama(env,
                 num,
                 model,
                 goals,
                 log=False,
                 forward_only=False,
                 classes=None):
    device = get_device(model)
    rotations = [
        hutil.quat_from_angle_axis(a, np.array([0, 1, 0]))
        for a in np.linspace(0, 2 * np.pi, endpoint=False, num=num)
    ]
    pos, rot = env.agent_state()
    ims = []
    dists = min_dists(env, goals)
    cols = 224
    scale = (4.0 / num) - 0.05
    rng = int(scale * cols / 2)
    for r in rotations:
        env.set_agent_state(pos, r * rot)
        obs = env.get_observation()['rgb']
        vals = model(numpy_to_imgnet(obs).unsqueeze(0).to(device))
        if forward_only:
            max_vals = vals[0, :, 0].detach().cpu().numpy()
        else:
            max_vals = vals.max(axis=2).values.squeeze().detach().cpu().numpy()
        env.step(0)
        dist_diff = -(min_dists(env, goals) - dists)
        im = obs[0] if len(obs.shape) == 4 else obs
        im = im[:, (cols // 2) - rng:(cols // 2) + rng, :]
        # im[:,-1] = 0
        if log:
            max_vals = np.log(max_vals)
        ims.append((im, max_vals, dist_diff))

    im, max_vals, dist_diff = util.unzip(reversed(ims))
    joined = np.concatenate(im, axis=1)
    fig, axes = plt.subplots(6,
                             1,
                             gridspec_kw={
                                 'hspace': 0,
                                 "wspace": 0,
                                 'height_ratios': [6, 0.5, 0.5, 0.5, 0.5, 0.5]
                             })
    fig.subplots_adjust(hspace=0, wspace=0)
    imax = axes[0]
    # imax.axis('on')
    pltaxes = axes[1:]
    imax.set_xlim((0,joined.shape[1]))
    imax.set_ylim((joined.shape[0],0))

    # search for right hiehgt
    low,high = 8,9
    for _ in range(20):
        mid = (high+low)/2
        fig.set_figheight(mid)
        fig.canvas.draw()
        imwidth = imax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        axwidth = pltaxes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        axwidth = pltaxes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        if imwidth == axwidth:
            print('eq')
            high = mid
        else:
            print('low')
            low = mid

    fig.set_figheight(high)
    fig.savefig('vis/test.pdf',bbox_inches='tight',pad_inches=0.0)

    imax.axis('on')
    imax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    imax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        labelleft=False)  # labels along the bottom edge are off

    vals = np.stack(max_vals).transpose()
    for ax, va in zip(pltaxes, vals):
        ax.imshow(va[None, :],
                  extent=[0, 12, 0, 1],
                  aspect='auto',
                  cmap='Wistia')
        ax.set_xlim((0, 12))
        ax.set_ylim((0,1))
        for i, v in enumerate(va):
            ax.text(i + 0.5,
                    0.45,
                    '%0.2f' % (v),
                    fontdict={'size': 16},
                    horizontalalignment='center',
                    verticalalignment='center')
        ax.axis('on')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        ax.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            labelleft=False)  # labels along the bottom edge are off

    ratio = joined.shape[0] / joined.shape[1]
    imax.imshow(joined)
    # fig.text(0.12, 0.41, 'Bed', horizontalalignment='right', fontdict={'size': 16})
    # fig.text(0.12, 0.34, 'Chair', horizontalalignment='right', fontdict={'size': 16})
    # fig.text(0.12, 0.271, 'Couch', horizontalalignment='right',
             # fontdict={'size': 16})
    # fig.text(0.12, 0.205, 'D. Table', horizontalalignment='right',
             # fontdict={'size': 16})
    # fig.text(0.12, 0.132, 'Toilet', horizontalalignment='right',
             # fontdict={'size': 16})
    # ratio = joined.shape[1] / (joined.shape[0] * 11 / 6)
    ratio = joined.shape[1] / (joined.shape[0] * 8.5 / 6)

    height = 8
    # searched for right high to deal with text shift
    # searched_height = 8.051948547363281
    # fig.set_figheight(searched_height)
    fig.set_figheight(height)
    fig.set_figwidth(height * ratio)
    fig.savefig('vis/test1.pdf',bbox_inches='tight',pad_inches=0.0)
    import pdb; pdb.set_trace()


    # set the agent back to where it was before
    env.set_agent_state(pos, rot)

    ims = list(reversed(ims))
    ims = [(im, val, cor) for im, val, cor in ims]
    annotated = []
    dists = np.array([[v, d] for _, v, d in ims])
    corrs = np.array(
        [np.corrcoef(dists[:, 0, i], dists[:, 1, i])[0, 1] for i in range(5)])
    return fig, corrs


if __name__ == '__main__':
    from experiment_config import ExperimentConfig
    # config = ExperimentConfig('logs/gibson_medium_noisy_inverse', device='cuda:6')
    gen = False
    rerender = True
    config = ExperimentConfig('logs/40k/detection_rewards_nopan',
                              device='cuda:6')
    from train_q_network import load_model_number
    model = load_model_number(config, 300000)
    model.eval()
    prefix = 'vis/pano_out'
    os.system(f'mkdir {prefix}')

    def gen_poses(env, floor):
        env.reset(floor)
        return env.pos, env.rot

    prefix = 'vis/pano'
    os.system(f'mkdir {prefix}')
    # sampling for interesting points
    # houses = [("Darden", 0), ('Markleeville', 1), ("Merom", 1)]
    # houses = [("Darden", 0), ("Darden", 1), ("Darden", 2), ('Corozal', 0),
              # ('Corozal', 1), ('Collierville', 0), ('Collierville', 2),
              # ('Collierville', 1), ('Markleeville', 1), ('Markleeville', 0),
              # ('Wiconisco', 0), ('Wiconisco', 1)]

    houses = [('Markleeville', 1,0)]
    if gen:
        poses = {}
        for house_name, floor in houses:
            house = gibson_info.get_house(house_name)
            env = house.get_env(
                torchmode=False,
                random_goal=True,
                panorama=config.PANORAMA,
                num_floors=house.num_floors,
                config_path='configs/tasks/pointnav_high_res.yaml')
            poses[(house_name, floor)] = np.array(
                [gen_poses(env, floor) for _ in range(500)])
            env.close()
        np.save('data_dump/pan_vis_poses', poses)
        exit()
    else:
        poses = np.load('data_dump/pan_vis_poses.npy', allow_pickle=True)[()]

    print(poses.keys())
    
    # houses = [("Merom",1)]
    # houses = [('Markleeville',1)]
    # houses = [("Merom",1)]
    houses = [('Corozal',0,325),('Markleeville',1,23),('Darden', 0,396),('Wiconisco',1,11),('Corozal',0,391),('Wiconisco',1,92)]
    # for house_name, floor in houses:
    num = 0
    for house_name, floor,ind in houses:
        house = gibson_info.get_house(house_name)
        env = house.get_env(torchmode=False,
                            random_goal=True,
                            panorama=config.PANORAMA,
                            num_floors=house.num_floors)

        env.reset(floor)
        locs = house.object_locations_for_habitat_dest
        goals = [
            gibson_info.relevant_locations(env.agent_state()[0], locs[k])
            for k in sorted(locs.keys())
        ]

        ims = []
        corrs = []
        positions = []

        top_down_map = maps.get_topdown_map(env.env.sim,
                                            map_resolution=(map_resolution,
                                                            map_resolution),
                                            draw_border=False)
        points = np.argwhere(top_down_map == 1)
        hposes = poses[(house_name, floor)]

        if rerender:
            pos,rot = hposes[ind]
            env.set_agent_state(pos, rot)
            im, cor = vis_panorama(env, 12, model, goals)
            im.savefig(f'vis/pano/final/{num}.pdf', bbox_inches='tight', pad_inches=0.05)
            num += 1
            env.close()
            continue

        for pos, rot in tqdm(hposes):
            env.set_agent_state(pos, rot)
            im, cor = vis_panorama(env, 12, model, goals)
            ims.append(im)
            corrs.append(cor)

        corrs = np.stack(corrs)
        for i in range(5):
            # sort from highest to lowest correlation
            order = (-corrs[:, i]).argsort()
            ordered_ims = [ims[o] for o in order]
            ordered_corrs = corrs[order.argsort(), i]
            top3 = ordered_ims[:30]
            for subi, im in enumerate(top3):
                path = f'{prefix}/{house.name}{floor}/class{i}-top{subi}-{order[subi]}.pdf'
                util.ensure_folders(path)
                im.savefig(path, bbox_inches='tight', pad_inches=0.05)

        print("dist graph")
        env.close()

    # points = [(np.array([-6.4714074, -2.674326,   1.4258186]), hutil.quat_from_coeffs([ 0.,         -0.97549421,  0.,         -0.22002511])),
    # (np.array([-10.189712,    -2.674326,    -0.61645865]), hutil.quat_from_coeffs([0.         ,0.73479027 ,0.,         0.67829436]))]

    # house = gibson_info.get_house("Pomaria")
    # floor =  0
    # locs = house.object_locations_for_habitat_dest
    # goals = [locs[k] for k in sorted(locs.keys())]
    # env = house.get_env(torchmode=False,random_goal=True,panorama=True,num_floors=house.num_floors)
    # for i,v in enumerate(points):
    # env.reset(0)
    # point,rot = v
    # env.set_agent_state(point,rot)
    # im,corrs = vis_panorama(env,12,model,goals)
    # print(corrs)
    # cv2.imwrite(f'vis/{i}.jpg', util.cv2.transform_rgb_bgr(im))
