from gibson_info import class_labels
import numpy as np
import torch
from torch.utils import data
import util
import util.habitat as habutil
from gibson_info import get_houses_medium, class_labels, get_houses, get_house, relevant_locations, relevant_objects
from matplotlib import pyplot as plt
from habitat.utils.visualizations import maps
import habitat_sim.utils.common as hutil
from tqdm import tqdm
import random
import queue
import os
import math
from visualize_panorama import join_images
from slam import DepthMapperAndPlanner
from evaluation.policy_defaults import name_from_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import cv2

def to_angle(rot):
    agent_rot = hutil.quat_to_angle_axis(rot)
    # sometimes? reports 2pi rotation around x axis if no rotation
    if agent_rot[1][1] == 0:
        return 0
    else:
        return (agent_rot[0] * agent_rot[1][1]) % (2 * np.pi)


def get_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(
        'configs/detectron/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    )
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    return DefaultPredictor(cfg)


def degree_to_rad(deg):
    return deg / 180 * np.pi

map_resolution = 1500
success_distance = 1
MAX_STEPS = 500

NUM_ROTATIONS = 12
# angs = np.linspace(0, 2 * np.pi, num=NUM_ROTATIONS, endpoint=False)
angs = np.linspace(0, 2 * np.pi, num=NUM_ROTATIONS + 1)[1:]
rotations = [hutil.quat_from_angle_axis(a, np.array([0, 1, 0])) for a in angs]

# Quat is reference to pointing toward negative z direction
# i.e. 1,0,0,0 -> -z and rotation angle is counter clockwise around the y axis, towards negative x
def check_movement(config, env, start_ang, planner=None, rng=random):
    points = []
    for _ in range(100):
        dist = rng.uniform(0.9, 2)
        ang = rng.uniform(-degree_to_rad(7), degree_to_rad(7)) + start_ang
        translation = np.array([-math.sin(ang), 0, -math.cos(ang)]) * dist
        points.append(translation + env.pos)

    if len(points) == 0: return None
    point_index = planner.reachable_nearby(points)
    if point_index is not None:
        return points[point_index]
    return None


# get scores fot the given class from MaskRCNN
def get_scores(predictor, im, class_index):
    predictions = predictor(im)
    pr = predictions['instances']
    mask = pr.pred_classes == class_index
    boxes = pr.pred_boxes.tensor[mask]
    scores = pr.scores[mask]
    return boxes, scores


# draw box from detector
def draw_box(im, box, score):
    rect = cv2.rectangle(im.copy(), tuple(box[0:2]), tuple(box[2:4]),
                         (255 * score, 0, 0), 3)
    # dealing with cv2 bug
    if rect.__class__ == np.ndarray:
        return rect
    else:
        return rect.get()


# Main function running the evaluation
def ours_evaluate(config, env, ep, house, epind, model, visualize,
                  model_config):
    hn, floor, class_label, goal_dist, pos, rot = ep

    if config.SCORE == 'detector' or config.COMBINE_DETECTOR:
        predictor = get_predictor()
        predictor_class_index = predictor.metadata.thing_classes.index(
            class_label)

    rng = random.Random()
    rng.seed(config.SEED)

    if goal_dist == float('inf'):
        return np.array([]) if config.STOP else 0

    class_index = class_labels.index(class_label)

    def model_score(ims):
        torch_ims = util.torch.to_imgnet(torch.tensor(ims).to('cuda'))
        with torch.no_grad():
            return model(torch_ims.unsqueeze(0))[0,
                                                 class_index, :].max().item()

    # compute the frame score combined with detector
    def score(ims):
        sc = model_score(ims['rgb'])
        if config.COMBINE_DETECTOR:
            size = ims['rgb'].shape[1]
            left_lim, right_lim = int(size / 3), int(size * 2 / 3)
            im = ims['rgb'][0] if len(ims['rgb']) == 4 else ims['rgb']
            boxes, scores = get_scores(predictor, im, predictor_class_index)
            if len(scores) > 0 and scores.max() > config.CONFIDENCE_THRESHOLD:
                box = boxes[scores.argmax()]
                if box[0] <= right_lim or box[2] >= left_lim:
                    if len(ims['rgb']) == 4:
                        ims['rgb'][0] = draw_box(ims['rgb'][0], box,
                                                 scores.max().item())
                    else:
                        ims['rgb'] = draw_box(ims['rgb'], box,
                                              scores.max().item())
                    sc += (scores.max().item() + 1)
        return sc

    def output():
        print(f"SPL: {spl}: {goal_dist}/{dist_traveled}")
        if config.SLAM and visualize:
            planner.write_combined(
                f'%04d_{class_label}-%dm-spl%.2f-steps%d' %
                (epind, int(goal_dist), spl, agent_steps_taken))
        # np.save(f'data_dump/good_trajectory{epind}',np.array(log))
        return np.array(log) if config.STOP else spl

    locs = house.object_locations_for_habitat_dest
    all_goals = [locs[k] for k in sorted(locs.keys())]

    from habitat.utils.visualizations import maps
    top_down_map = maps.get_topdown_map(env.env.sim,
                                        map_resolution=(map_resolution,
                                                        map_resolution))
    rrange, crange = util.habitat.crop_range(top_down_map)
    point_min = util.habitat.from_grid([rrange[0], crange[0]], map_resolution,
                                       0)
    point_max = util.habitat.from_grid([rrange[1], crange[1]], map_resolution,
                                       0)
    max_dim = np.abs(point_max - point_min).max()

    out_dir = f'{config.VIDEO_LOCATION}/{name_from_config(config)}'
    util.ensure_folders(out_dir, True)
    planner = DepthMapperAndPlanner(dt=30,
                                    out_dir=out_dir,
                                    map_size_cm=max_dim * 230,
                                    mark_locs=True,
                                    close_small_openings=True,
                                    log_visualization=visualize)
    polygons = relevant_objects(env.pos, house.objects[class_label])
    planner._reset(goal_dist,
                   global_goals=polygons,
                   start_pos=env.pos,
                   start_ang=env.angle)

    openlist = []
    visited = []
    dist_traveled = 0
    log = []
    spl = 0
    agent_steps_taken = 0
    full_log = []
    episode_ims = [env.get_observation()]
    no_move = False

    def semantic_reasoning():
        # log for visualizations
        planner.log_reasoning()

        images = []
        display_values = []
        for _ in range(NUM_ROTATIONS):
            ims, _, _, _ = env.step(1)
            loc = [*planner.pos_to_loc(env.pos), env.angle]
            planner.add_observation(ims['depth'] * 1000, loc)
            dest = check_movement(config,
                                  env,
                                  env.angle,
                                  planner=planner,
                                  rng=rng)
            sc = score(ims)
            # for visualizations
            images.append(ims)
            display_values.append(sc)
            if dest is not None:
                openlist.append((sc, dest))

        if visualize and config.SLAM:
            ims_to_render = [
                e['rgb'][0] if len(e['rgb'].shape) == 4 else e['rgb']
                for e in images
            ]
            current_pan = join_images(
                ims_to_render,
                -np.array(display_values),
                bl_text='Predicted Values',
                br_text=f'Object Class: {class_label.title()}')
            planner.set_current_pan(current_pan)

    macro_steps = 50 if config.SLAM else 30
    print("goals ", env.goals)

    # initial steps to scan env and choose dest
    semantic_reasoning()
    agent_steps_taken += NUM_ROTATIONS

    for macro_step_num in range(macro_steps):
        print(agent_steps_taken)

        if config.BACKTRACK_REJECTION and len(visited) > 0:
            vis_stack = np.stack(visited)

            def reject(point):
                dists = np.linalg.norm((vis_stack - point)[:, [0, 2]], axis=1)
                return (dists < (success_distance - 0.1)).sum() > 0

            openlist = [e for e in openlist if not reject(e[1])]

        def maxfunc(x):
            s, d = x
            dist = np.linalg.norm(env.pos - d)
            return s + config.CONSISTENCY_WEIGHT * max(10 - dist, 0) / 10

        if len(openlist) == 0:
            print("openlist exhausted")
            if visualize: planner.write_combined()
            return output()

        ind, (sc, next_pos), _ = util.argmax(openlist, maxfunc)
        openlist.pop(ind)

        original_openlist = openlist.copy()

        # remove points which we cannot move toward, with an exception
        # for the first step due to the initilization process
        dist_est = planner.fmmDistance(next_pos)
        while not planner.action_toward(next_pos):
            if len(openlist) == 0:
                print("openlist exhausted fmm")
                if visualize: planner.write_combined()
                return output()
            ind, (sc, next_pos), _ = util.argmax(openlist, maxfunc)
            openlist.pop(ind)
            dist_est = planner.fmmDistance(next_pos)

        print('score of', sc)

        if visualize and config.SLAM:
            planner.set_current_open(openlist)

        obs = env.get_observation()
        planner.set_goal(next_pos)
        goal_reached = False

        # 6 for initial rotation, and 2x distance*4 to
        # account for 1 rotation per forward step on average
        step_estimate = math.ceil(2 * (dist_est / 0.25) + 6)
        cur_dist_est = dist_est
        for step in range(step_estimate):
            new_dist_est = planner.fmmDistance(next_pos)
            # replan if the estimated distance jumps up too much
            if new_dist_est > cur_dist_est + 0.1:
                print('replan')
                break
            cur_dist_est = new_dist_est
            action = planner.get_action_toward(next_pos)
            print('action: ', action)

            if action == 3:
                print('subgoal reached')
                break

            before_pos = env.pos
            obs, _, _, _ = env.step(action)

            if action == 0:
                dist_traveled += 0.25
            planner.pos_to_loc(env.pos)
            planner.log_act(obs, env.pos, env.angle, action)
            episode_ims.append(obs)
            visited.append(env.pos)
            log.append([
                env.pos, env.rot, dist_traveled,
                env.distance_to_goal(), step == 0
            ])
            agent_steps_taken += 1

            if env._dist_to_goal(
                    env.pos) < success_distance and not config.STOP:
                spl = min(goal_dist / (dist_traveled + 1e-5), 1)
                return output()
            if agent_steps_taken >= MAX_STEPS: return output()
        semantic_reasoning()
        agent_steps_taken += NUM_ROTATIONS
        if agent_steps_taken >= MAX_STEPS: return output()
    return output()
