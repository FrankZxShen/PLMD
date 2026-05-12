from collections import deque, defaultdict
from typing import Dict
from itertools import count
import os
import logging
import time
import json
import sys
import gym
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import quaternion
import pickle
import io
import re
from copy import deepcopy
from torchvision import transforms

from skimage import measure
from skimage.color import gray2rgb, rgb2gray
from skimage.feature import canny
import skimage.morphology
from PIL import Image

import math
import cv2
import habitat
import habitat_sim
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from utils.hdbscan_utils import HdbscanCluster
from nav_utils.goal_find_utils import find_navigation_target, add_text_with_rounded_rectangle

def tensor_to_image():

    return transforms.ToPILImage()

def image_to_tensor():

    return transforms.ToTensor()

def calculate_distance(coord1, coord2):

    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def find_content_bbox(image, threshold=30, white_threshold=245):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, gray.mean())
    
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    white_mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(white_mask))
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0, image.shape[1], image.shape[0]

    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    return x, y, x+w, y+h

def smart_crop_resize(img):

    x1, y1, x2, y2 = find_content_bbox(img)
    
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    size = max(crop_width, crop_height)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    new_x1 = max(0, center_x - size//2)
    new_y1 = max(0, center_y - size//2)
    new_x2 = min(img.shape[1], new_x1 + size)
    new_y2 = min(img.shape[0], new_y1 + size)
    
    if new_x2 - new_x1 != new_y2 - new_y1:
        size = min(new_x2 - new_x1, new_y2 - new_y1)
        new_x2 = new_x1 + size
        new_y2 = new_y1 + size


    cropped = img[new_y1:new_y2, new_x1:new_x2]
    a=[new_y1,new_y2,new_x1,new_x2]

    resized = cv2.resize(cropped, (256, 256), 
                        interpolation=cv2.INTER_LANCZOS4)
    
    return a,resized


# from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
# from constants import coco_categories, color_palette, category_to_id
from agents.panorama_vlm_agents import LLM_Agent
from agents.panorama_vlm_agents_gt import LLM_Agent_GT
# from agents.llm_agents import LLM_Agent
from constants import (
    color_palette, coco_categories, coco_categories_hm3d2mp3d,
    hm3d_category, category_to_id, object_category
)
from envs.habitat.multi_agent_env_vlm import Multi_Agent_Env

from src.vlm import CogVLM2
from src.SystemPrompt import (
    form_prompt_for_PerceptionVLM, 
    form_prompt_for_FN,
    form_prompt_for_DecisionVLM_Frontier_COT1,
    form_prompt_for_DecisionVLM_Frontier_COT2,
    form_prompt_for_DecisionVLM_History,

    form_prompt_for_DecisionVLM_MetaPreprocess,
    form_prompt_for_Module_Decision,
    Perception_weight_decision,
    Perception_weight_decision4,
    Perception_weight_decision26,
    extract_scene_image_description_results,
    extract_scene_object_detection_results,
    extract_scenario_exploration_analysis_results
)
# from src.tsdf import TSDFPlanner
import utils.pose as pu

import utils.visualization as vu

from arguments import get_args

from detect.ultralytics import YOLOv10

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


@habitat.registry.register_action_space_configuration
class PreciseTurn(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.TURN_LEFT_S] = habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )
        config[HabitatSimActions.TURN_RIGHT_S] = habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )

        return config


def Objects_Extract(args, full_map_pred, use_sam):

    semantic_map = full_map_pred[4:]

    dst = np.zeros(semantic_map[0, :, :].shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    Object_list = {}
    for i in range(len(semantic_map)):
        if semantic_map[i, :, :].sum() != 0:
            Single_object_list = []
            se_object_map = semantic_map[i, :, :].cpu().numpy()
            se_object_map[se_object_map>0.1] = 1
            se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if len(cnt) > 30:
                    epsilon = 0.05 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    Single_object_list.append(approx)
                    cv2.polylines(dst, [approx], True, 1)
            if len(Single_object_list) > 0:
                # print(i)
                # print(Single_object_list)
                if use_sam:
                    Object_list[object_category[i]] = Single_object_list
                else:
                    if 'objectnav_mp3d' in args.task_config:
                        Object_list[object_category[i]] = Single_object_list
                    elif 'objectnav_hm3d' in args.task_config:
                        if i >= 15:
                            pass
                        else:
                            Object_list[hm3d_category[i]] = Single_object_list
    return Object_list

def all_agents_exit_false(agents):
    for agent in agents:
        if agent.EXIT:
            return False
    return True

def all_agents_exit_true(agents):
    for agent in agents:
        if not agent.EXIT:
            return False
    return True


def get_Frontiers(full_map_pred):
    # ------------------------------------------------------------------
    ##### Get the frontier map and filter
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    full_w = full_map_pred.shape[1]
    local_ex_map = np.zeros((full_w, full_w))
    local_ob_map = np.zeros((full_w, full_w))

    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)

    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(),0.1,1)
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(local_ex_map,contour,-1,1,1)

    # clear the boundary
    local_ex_map[0:2, 0:full_w]=0.0
    local_ex_map[full_w-2:full_w, 0:full_w-1]=0.0
    local_ex_map[0:full_w, 0:2]=0.0
    local_ex_map[0:full_w, full_w-2:full_w]=0.0

    target_edge = local_ex_map-local_ob_map
    # print("local_ob_map ", self.local_ob_map[200])
    # print("full_map ", self.full_map[0].cpu().numpy()[200])

    target_edge[target_edge>0.8]=1.0
    target_edge[target_edge!=1.0]=0.0

    wall_edge = local_ex_map - target_edge

    # contours, hierarchy = cv2.findContours(cv2.inRange(wall_edge,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)>0:
    #     dst = np.zeros(wall_edge.shape)
    #     cv2.drawContours(dst, contours, -1, 1, 1)

    # edges = cv2.Canny(cv2.inRange(wall_edge,0.1,1), 30, 90)
    Wall_lines = cv2.HoughLinesP(cv2.inRange(wall_edge,0.1,1), 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=10)

    # original_image_color = cv2.cvtColor(cv2.inRange(wall_edge,0.1,1), cv2.COLOR_GRAY2BGR)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    img_label, num = measure.label(target_edge, connectivity=2, return_num=True)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等

    Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
    Goal_point = []
    Goal_area_list = []
    dict_cost = {}
    for i in range(1, len(props)):
        if props[i].area > 4:
            dict_cost[i] = props[i].area

    if dict_cost:
        dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

        for i, (key, value) in enumerate(dict_cost):
            Goal_edge[img_label == key + 1] = 1
            Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
            Goal_area_list.append(value)
            if i == 3:
                break
        # frontiers = cv2.HoughLinesP(cv2.inRange(Goal_edge,0.1,1), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

        # original_image_color = cv2.cvtColor(cv2.inRange(Goal_edge,0.1,1), cv2.COLOR_GRAY2BGR)
        # if frontiers is not None:
        #     for frontier in frontiers:
        #         x1, y1, x2, y2 = frontier[0]
        #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return Wall_lines, Goal_area_list, Goal_edge, Goal_point

def Visualize(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge, Frontiers_dict, goal_points, \
              is_labeled, is_dif, sem_map_labeled, sem_map_diffusion):
    
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, episode_n)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    # no_cat_mask = sem_map == 20
    if 'objectnav_hm3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2
    elif 'objectnav_mp3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2 + 5
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    if 'objectnav_mp3d' in args.task_config:
        cn = goal_name + 4
    elif 'objectnav_hm3d' in args.task_config:
        cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents and goal_points[i][0] != 9999:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i
    
    # 画出全局语义地图
    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    # vis_image = vu.init_multi_vis_image(category_to_id[goal_name], color)
    if 'objectnav_mp3d' in args.task_config:
        vis_image = vu.init_multi_diffusion_vis_image(object_category[goal_name], color)
    elif 'objectnav_hm3d' in args.task_config:
        vis_image = vu.init_multi_diffusion_vis_image(object_category[coco_categories_hm3d2mp3d[goal_name]], color)

    # vis_image[50:530, 15:495] = sem_map_vis
    vis_image[50:530, 15:495] = sem_map_vis
    if is_labeled:
        vis_image[50:530, 510:990] = sem_map_labeled
    
    if is_dif:
        vis_image[50:530, 1005:1485] = sem_map_diffusion
        vis_image[50, 1005:1485] = [100,100,100]

    for i in range(args.num_agents):
        agent_arrow = vu.get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)
    
    
    if args.print_images:
        fn = '{}/episodes/eps_{}/Step-{}.png'.format(
            dump_dir, episode_n,
            l_step)
        # print(fn)
        cv2.imwrite(fn, vis_image)   


def semantic_map_vis_to_pred(sem_map_vis, color_palette, threshold=30):
    sem_map_vis = cv2.cvtColor(sem_map_vis, cv2.COLOR_BGR2RGB)
    sem_map_vis = Image.fromarray(sem_map_vis)

    
    # color_palette是一个大小120的列表，我需要你将其按照每三个元素将其划分为40*3的颜色矩阵，每个元素代表一个颜色通道
    color_pal = [int(x * 255.) for x in color_palette]
    color_palette = np.array(color_pal).reshape(-1, 3)
    color_to_label = {tuple(color): i for i, color in enumerate(color_palette)}
    sem_map = np.zeros(sem_map_vis.size[::-1], dtype=np.uint8)
    
    rgb_data = np.array(sem_map_vis) #[480,480,3]
 
    for y in range(rgb_data.shape[0]):
        for x in range(rgb_data.shape[1]):
            pixel = tuple(rgb_data[y, x])
            min_dist = 10
            best_label = 0
            for color, label in color_to_label.items():
                dist = sum((a-b)**2 for a, b in zip(pixel, color))
                if dist < min_dist:
                    min_dist = dist
                    best_label = label
            if min_dist < threshold:  
                sem_map[y, x] = best_label
    
    sem_map = np.flipud(sem_map)

    total_channels = len(color_palette)
    full_map_pred = np.zeros((total_channels, sem_map.shape[0], sem_map.shape[1]), dtype=np.float32)
    for i in range(total_channels - 4): 
        full_map_pred[i + 4] = (sem_map == i).astype(np.float32)
    
    return torch.from_numpy(full_map_pred)


def Decision_Generation_Vis(args, sde, S_sde, model, agent, agents_seg_list, agent_j, episode_n, l_step, pose_pred, full_map_pred, goal_name,
                             visited_vis, map_edge, history_nodes, Frontiers_dict, goal_points, pre_goal_point):
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, episode_n)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    # no_cat_mask = sem_map == 20
    if 'objectnav_hm3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2
    elif 'objectnav_mp3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2 + 5
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3

    # Del Traj
    sem_map[sem_map == 3] = 2
    # Del long-time goal
    sem_map[sem_map == 4] = 2


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    if 'objectnav_mp3d' in args.task_config:
        cn = goal_name + 4
    elif 'objectnav_hm3d' in args.task_config:
        cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents and goal_points[i][0] != 9999:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i

    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)
    
    sem_map_diffusion = deepcopy(sem_map)
    
    flag_diffusion = False
    Diffusion_Frontiers_dict = {}
    diffusion_agents_seg_list = None
################################################################################# PLMD ###################################################################
##########
    sem_map_vis3 = None
    if l_step > 95 and (l_step+1) % 50 == 0 and agent.Find_Goal== 0 and agent_j + 1 == args.num_agents:
        flag_diffusion = True
        pos = pose_pred[agent_j]
        transform = transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        # diffusion
        sem_map_diffusion_vis = Image.new("P", (sem_map_diffusion.shape[1],
                                        sem_map_diffusion.shape[0]))
        sem_map_diffusion_vis.putpalette(color_pal)
        sem_map_diffusion_vis.putdata(sem_map_diffusion.flatten().astype(np.uint8))
        sem_map_diffusion_vis = sem_map_diffusion_vis.convert("RGB")
        sem_map_diffusion_vis = np.flipud(sem_map_diffusion_vis)

        sem_map_diffusion_vis = sem_map_diffusion_vis[:, :, [2, 1, 0]]
        sem_map_diffusion_vis = cv2.resize(sem_map_diffusion_vis, (480, 480),
                                    interpolation=cv2.INTER_NEAREST)
        print("TimeStep {}, local map diffusion process...".format(l_step))

        sem_map_vis_tmp = deepcopy(sem_map_diffusion_vis)
        crop_size,sem_map_diffusion_vis = smart_crop_resize(cv2.cvtColor(sem_map_diffusion_vis, cv2.COLOR_RGB2BGR))
        x1, x2, y1, y2 = crop_size
        target_size = x2 - x1

        mask = np.full(sem_map_diffusion_vis.shape[:2], 255, dtype=np.uint8)
        white_threshold = np.array([245, 245, 245])
        white_mask = np.all(sem_map_diffusion_vis > white_threshold, axis=-1)
        mask[white_mask] = 0 
        
        mask = transform(Image.fromarray(mask))
        mask = mask.unsqueeze(0)

        sem_map_diffusion_vis_n = sem_map_diffusion_vis.astype(np.float32) / 255.
        gray_image = rgb2gray(np.array(tensor_to_image()(sem_map_diffusion_vis_n)))
        edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=2.)))
        gray_image = image_to_tensor()(Image.fromarray(gray_image))
        Y_GT, X_GT, X_LQ = sem_map_diffusion_vis_n,gray_image,edge ##completed grayscale and edge images

        # transform = transforms.Compose([
        #     transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
        #     transforms.ToTensor(),
        # ])

        Y_GT = torch.from_numpy(Y_GT)
        Y_GT = Y_GT.permute(2, 0, 1).unsqueeze(0)

        # # save gt
        # print(Y_GT.shape)#[1,3,256,256]
        # print(mask.shape)#[1,1,256,256]
        # Y_GT_mask = Y_GT*mask

        # # print(Y_GT.squeeze(0).permute(1,2,0).numpy().shape)

        # save_img_path = '/home/szx/project/PEANUT/data/Nav_test_result/GT.png'
        # util.save_img(Y_GT.squeeze(0).permute(1,2,0).numpy()*255, save_img_path)

        # # save mask+gt(S_{mask})
        # save_img_path = '/home/szx/project/PEANUT/data/Nav_test_result/GT+mask.png'
        # util.save_img(Y_GT_mask.squeeze(0).permute(1,2,0).numpy()*255, save_img_path)


        noisy_state = sde.noise_state(Y_GT * mask) # *mask
        noisy_states = S_sde.noise_state(X_LQ * mask) # * mask
        model.feed_data(noisy_state, Y_GT * mask, Y_GT, mask, S_sde, X_GT,  X_LQ * mask)
        model.test(sde, save_states=True, GT = Y_GT, mask = mask, \
                        S_sde = S_sde, S_GT = X_GT, S_LQ = noisy_states, dis = model.dis, save_dir=None)
        
        # toc = time.time()#
        # test_times.append(toc - tic)
        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())  # uint8
        # LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        # GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8

        resized_output = cv2.resize(output, (target_size, target_size), 
                            interpolation=cv2.INTER_LANCZOS4)
        sem_map_vis_tmp[x1:x2,y1:y2] = resized_output
        
        output = cv2.resize(sem_map_vis_tmp, (480, 480), interpolation=cv2.INTER_NEAREST)
        output_restore = deepcopy(output)


        # restore full_map_pred
        restored_full_map_pred = semantic_map_vis_to_pred(output, color_palette)
        

################################################################
        sem2 = restored_full_map_pred[4:, :,:].argmax(0).cpu().numpy()
        sem_map_vis3 = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
        sem_map_vis3.putpalette(color_pal)
        sem_map_vis3.putdata(sem2.flatten().astype(np.uint8))
        sem_map_vis3 = sem_map_vis3.convert("RGB")
        sem_map_vis3 = np.flipud(sem_map_vis3)

        sem_map_vis3 = sem_map_vis3[:, :, [2, 1, 0]]
        sem_map_vis3 = cv2.resize(sem_map_vis3, (480, 480),
                                    interpolation=cv2.INTER_NEAREST)
        
        # sem_map_path = f"/home/szx/project/Co-NavGPT/data/VLM_EXP/restored_semmap.png"
        # cv2.imwrite(sem_map_path, sem_map_vis3)
        vis_image = vu.init_multi_diffusion_vis_image_back()

        # vis_image[50:530, 15:495] = sem_map_vis
        
    


########################################===Localization Strategy===########################################
        color_pal_resize = np.array(color_pal).reshape(-1, 3)
        # global_goals = np.argwhere(output == color_pal_resize[self.goal_cat+5]) # goal_cat是基于map_category_names的映射  
        if 'objectnav_hm3d' in args.task_config:
            matches = np.where((output == color_pal_resize[coco_categories[agent.goal_id] + 5]).all(axis=2))
        elif 'objectnav_mp3d' in args.task_config:
            matches = np.where((output == color_pal_resize[agent.goal_id + 5]).all(axis=2))
        global_goals = []
        for y, x in zip(matches[0], matches[1]):
            global_goals.append([y, x])
        global_goals = np.array(global_goals)

        if len(global_goals) > 0:
            
            global_goal,num_points,cluster_density = HdbscanCluster().predict(global_goals)
            # global_goal = [goal[:-1] for goal in global_goal]
            
            for global_goal2 in global_goals:
                center2 = (global_goal2[1],global_goal2[0])
                # cv2.circle(local_sem_map, center2, radius=1, color=(255, 0, 0), thickness=-1) 
                cv2.circle(output, (global_goal2[1],global_goal2[0]), radius=1, color=(255, 0, 0), thickness=-1) 
            if len(global_goal) > 0:
                for center in global_goal:
                    local_center = (center[1], center[0])
                    # cv2.circle(local_sem_map, local_center, radius=5, color=(0, 0, 255), thickness=-1) 
                    cv2.circle(output, (center[1],center[0]), radius=5, color=(0, 0, 255), thickness=-1) 
            elif len(global_goals) > 0:
                distances = np.linalg.norm(global_goals - pos, axis=1)
                closest_index = np.argmin(distances)
                t_point = global_goals[closest_index]
                cv2.circle(output, (t_point[1],t_point[0]), radius=5, color=(0, 0, 255), thickness=-1)


            if len(global_goal)>0:
                fullmap_new_goal_point = find_navigation_target(np.array(global_goal),np.array(num_points),np.array(cluster_density), \
                                                                        gx1, gx2, gy1, gy2, pos)
                new_long_term_goal_point = (fullmap_new_goal_point[0],fullmap_new_goal_point[1])
                
                # for center in centers:
                #     cv2.circle(image, tuple(center), radius=5, color=(0, 0, 255), thickness=-1)
                new_pred_local_goal_map = np.zeros((480,480))
                new_pred_local_goal_map[int(new_long_term_goal_point[0]),int(new_long_term_goal_point[1])] = 1

                new_pred_goal_map = np.flipud(new_pred_local_goal_map)
            elif len(global_goals) > 0:
                fullmap_new_goal_point = tuple(t_point)
                new_long_term_goal_point = (fullmap_new_goal_point[0],fullmap_new_goal_point[1])
                
                # for center in centers:
                #     cv2.circle(image, tuple(center), radius=5, color=(0, 0, 255), thickness=-1)
                new_pred_local_goal_map = np.zeros((480,480))
                new_pred_local_goal_map[int(new_long_term_goal_point[0]),int(new_long_term_goal_point[1])] = 1

                new_pred_goal_map = np.flipud(new_pred_local_goal_map)
                


        # self.diffusion_output = local_sem_map
        diffusion_output_local = output

        vis_image[50:530, 15:495] = sem_map_vis 
        vis_image[50:530, 510:990] = sem_map_vis3 
        vis_image[50:530, 1005:1485] = diffusion_output_local 
        vis_image[50, 1005:1485] = [100,100,100]

        if args.print_images:
            fn_dir = '{}/episodes/eps_{}/SemMap'.format(
                dump_dir, episode_n)
            if not os.path.exists(fn_dir):
                os.makedirs(fn_dir)
            fn_path = '{}/episodes/eps_{}/SemMap/Step-{}.png'.format(
                dump_dir, episode_n,
                l_step)
            cv2.imwrite(fn_path, vis_image)
################################################################
        ##### 
        diffusion_agents_seg_list = Objects_Extract(args, restored_full_map_pred, args.use_sam)

        _, diffusion_Frontier_list, _, diffusion_target_point_map = get_Frontiers(restored_full_map_pred)

        if len(diffusion_target_point_map) > 0:
            for j in range(len(diffusion_target_point_map)):
                Diffusion_Frontiers_dict['Diffusion_frontier-' + str(j)] = f"<centroid: {diffusion_target_point_map[j][0], diffusion_target_point_map[j][1]}, number: {diffusion_Frontier_list[j]}>"
            logging.info(f'=====> Diffusion Frontier: {diffusion_Frontier_list}')

########################################====Draw Label===########################################
    # sem_map_vis #[480,480,3]

    color_black = (0,0,0)
    color_green = (0,255,0)
    color_red = (0,0,255)
    color_blue = (255,0,0)
    color_black_blue = (150,0,0)
    color_yellow = (255,0,255)
    pattern = r'<centroid: (.*?), (.*?), number: (.*?)>'
    alpha = [chr(ord("A") + i) for i in range(26)]
    alpha0 = 0
    
    def d240(x):
        if x < 240:
            x = x + 2*(240-x)
        elif x >= 240:
            x = x - 2*(x-240)
        return x

    # for i in range(args.num_agents):
    #     agent_arrow = vu.get_contour_points(pose_pred[i], origin=(0, 0), size=10)

    #     cv2.drawContours(sem_map_vis, [agent_arrow], 0, color[i], -1)
    # agent_arrow = vu.get_contour_points(pose_pred[agent_j], origin=(0, 0), size=10)

    # cv2.drawContours(sem_map_vis, [agent_arrow], 0, color[agent_j], -1)

    sem_map_vis2 = sem_map_vis.copy()
    ##### 
    for key, value in agents_seg_list.items():
        for array in value:
            pts = array.reshape((-1, 1, 2))
            if agent_j == 0:
                for i in pts:
                    for j in i:
                        j[1] = d240(j[1])
            
            x, y, w, h = cv2.boundingRect(pts)
            text_position = (x + w // 2, y + h // 2)

            sem_map_vis = add_text_with_rounded_rectangle(
                img=sem_map_vis,
                text=key,
                font_size=24,
                font_color=(0, 0, 0),
                bg_color=(203,192,255),
                padding=10,
                corner_radius=10,
                position=(text_position[0] - 30, text_position[1] - 15)
            )
            sem_map_vis2 = add_text_with_rounded_rectangle(
                img=sem_map_vis2,
                text=key,
                font_size=24,
                font_color=(0, 0, 0),
                bg_color=(203,192,255),
                padding=6,
                corner_radius=10,
                position=(text_position[0] - 30, text_position[1] - 15)
            )
                
    if flag_diffusion:
        for key, value in diffusion_agents_seg_list.items():
            for array in value:
                pts = array.reshape((-1, 1, 2))
                if agent_j != 0:
                    for i in pts:
                        for j in i:
                            j[1] = d240(j[1])
                x, y, w, h = cv2.boundingRect(pts)
                text_position = (x + w // 2, y + h // 2)

                sem_map_vis3 = add_text_with_rounded_rectangle(
                img=sem_map_vis3,
                text=key,
                font_size=24,
                font_color=(0, 0, 0),
                bg_color=(203,192,255),
                padding=6,
                corner_radius=10,
                position=(text_position[0] - 30, text_position[1] - 15)
                )
                # cv2.putText(sem_map_vis3, key, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)# diffusion图
        sem_map_vis3 = np.array(sem_map_vis3,dtype=np.uint8)
    sem_map_vis = np.array(sem_map_vis,dtype=np.uint8)
    sem_map_vis2 = np.array(sem_map_vis2,dtype=np.uint8)

    if Frontiers_dict:
        for keys, value in Frontiers_dict.items():
            match = re.match(pattern, value)
            if match:
                centroid_x = int(match.group(1)[1:])
                centroid_y = int(match.group(2)[:-1])
                number = float(match.group(3))
              
                cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_black, -1)
                if flag_diffusion:
                    cv2.circle(sem_map_vis3, (centroid_y, d240(centroid_x)), 5, color_black, -1) # diffusion
                label = f"{alpha[alpha0]}"
                alpha0 += 1
                cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 1)
                if flag_diffusion:
                    cv2.putText(sem_map_vis3, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 1)

    if flag_diffusion:
        if Diffusion_Frontiers_dict:
            for keys, value in Diffusion_Frontiers_dict.items():
                match = re.match(pattern, value)
                if match:
                    centroid_x = int(match.group(1)[1:])
                    centroid_y = int(match.group(2)[:-1])
                    number = float(match.group(3))
                    # print(f"Centroid: ({centroid_x}, {centroid_y})")
                    # print(f"Number: {number}")
                    cv2.circle(sem_map_vis3, (centroid_y, d240(centroid_x)), 5, color_black, -1) # diffusion
                    label = f"{alpha[alpha0]}"
                    alpha0 += 1
                    cv2.putText(sem_map_vis3, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black_blue, 1)

    beta = [chr(ord("a") + i) for i in range(26)]
    alpha0 = 0
    if len(history_nodes) > 0:
        for hs in history_nodes[:26]:
            centroid_x = int(hs[0])
            centroid_y = int(hs[1])
            cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_green, -1)
            if flag_diffusion:
                cv2.circle(sem_map_vis3, (centroid_y, d240(centroid_x)), 5, color_green, -1)# diffusion
            label = f"{beta[alpha0]}"
            alpha0 += 1
            cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 1)
            if flag_diffusion:
                cv2.putText(sem_map_vis3, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 1)# diffusion图
        alpha0 = 0
        for hs in history_nodes[26:]:
            centroid_x = int(hs[0])
            centroid_y = int(hs[1])
            cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_green, -1)
            if flag_diffusion:
                cv2.circle(sem_map_vis3, (centroid_y, d240(centroid_x)), 5, color_green, -1)# diffusion
            label = f"{alpha[alpha0]}"
            alpha0 += 1
            cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 1)
            if flag_diffusion:
                cv2.putText(sem_map_vis3, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 1)# diffusion图
    
    sem_map_labeled = deepcopy(sem_map_vis)
    sem_map_labeled_diffusion = deepcopy(sem_map_vis3)

    agent_arrow = vu.get_contour_points(pose_pred[agent_j], origin=(0, 0), size=15)
    cv2.drawContours(sem_map_vis, [agent_arrow], 0, color_red, -1)
    cv2.drawContours(sem_map_vis2, [agent_arrow], 0, color_red, -1)
    if flag_diffusion:
        cv2.drawContours(sem_map_vis3, [agent_arrow], 0, color_red, -1)# diffusion
    if pre_goal_point:
        cv2.circle(sem_map_vis, (int(pre_goal_point[1]), int(d240(pre_goal_point[0]))), 8, color_blue, -1)
        cv2.circle(sem_map_vis2, (int(pre_goal_point[1]), int(d240(pre_goal_point[0]))), 8, color_blue, -1)
        if flag_diffusion:
            cv2.circle(sem_map_vis3, (int(pre_goal_point[1]), int(d240(pre_goal_point[0]))), 8, color_blue, -1)# diffusion

    
    return sem_map_vis, sem_map_vis2, sem_map_vis3, Diffusion_Frontiers_dict, sem_map_labeled, sem_map_labeled_diffusion, diffusion_agents_seg_list, flag_diffusion




def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)



from PLMD import create_model
import PLMD.obst_utils as obst_util
#sys.path.insert(0, "../../")
import PLMD as util
import options as option

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:1" if args.cuda else "cpu")

    # logging.info(f"stride:{stride}")
    # logging.info(f"names:{names}")
    # logging.info(f"pt:{pt}")


    HabitatSimActions.extend_action_space("TURN_LEFT_S")
    HabitatSimActions.extend_action_space("TURN_RIGHT_S")

    config_env = habitat.get_config(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    config_env.defrost()
    
    agent_sensors = []
    agent_sensors.append("RGB_SENSOR")
    agent_sensors.append("DEPTH_SENSOR")
    agent_sensors.append("SEMANTIC_SENSOR")

    config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
    config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = \
        [0, args.camera_height, 0]

    config_env.TASK.POSSIBLE_ACTIONS = config_env.TASK.POSSIBLE_ACTIONS + [
        "TURN_LEFT_S",
        "TURN_RIGHT_S",
    ]
    config_env.TASK.ACTIONS.TURN_LEFT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_LEFT_S.TYPE = "TurnLeftAction_S"
    config_env.TASK.ACTIONS.TURN_RIGHT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_RIGHT_S.TYPE = "TurnRightAction_S"
    config_env.SIMULATOR.ACTION_SPACE_CONFIG = "PreciseTurn"
    config_env.freeze()

    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    model = create_model(opt) 
    device = model.device

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)
       
    S_sde = obst_util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    S_sde.set_model(model.models)

    # Load VLM
    # vlm = VLM(args.vlm_model_id, args.hf_token, device)
    base_url = args.base_url 
    cogvlm2 = CogVLM2(base_url) 
    # Load Yolo
    # yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)
    if args.yolo == 'yolov9':
        # yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)
        pass
    else:
        yolo = YOLOv10.from_pretrained(args.yolo_weights)
    # print(config_env)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # exit(0)


    
    env = Multi_Agent_Env(config_env=config_env)

    num_episodes = env.number_of_episodes

    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = config_env.SIMULATOR.NUM_AGENTS

    agent = []
    agent_GT = []
    for i in range(num_agents):
        agent.append(LLM_Agent(args, config_env, i, device, model=model, sde=None, S_sde=None,))
        if args.not_explore == 1:
            if 'objectnav_hm3d' in args.task_config:
                agent_GT.append(LLM_Agent_GT(args, config_env, i, device))

    # ------------------------------------------------------------------
    ##### Setup Logging
    # ------------------------------------------------------------------
    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        # filename=log_dir + 'output.log',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ])
    print("Dumping at {}".format(log_dir))
    # print(args)
    # logging.info(args)
    # ------------------------------------------------------------------

    # print("num_episodes:",num_episodes)# 1000

    agg_metrics: Dict = defaultdict(float)
    # obj_SR: Dict = defaultdict(float)
    # sys_metrics: Dict = defaultdict(float)
    agg_metrics['multi_Total_SR'] = 0
    agg_metrics['SPL'] = 0
    agg_metrics['SoftSPL'] = 0
    agg_metrics['multi_SPL'] = {}
    agg_metrics['multi_SoftSPL'] = {}
    agg_metrics['multi_Navigation_SR'] = 0
    for i in range(num_agents):
        agg_metrics['multi_SPL'][f'Agent_{i}'] = 0
        agg_metrics['multi_SoftSPL'][f'Agent_{i}'] = 0

    count_episodes = 0
    count_step = 0

    # diffusion
    count_DF = 0
    count_DFSR = 0

    goal_points = []
    
    log_start = time.time()
    last_decision = []
    total_usage = []

    history_nodes = []
    history_score = []
    history_count = []
    history_states = []

    cur_goal_points = []
    pre_goal_points = []

    # Map & Step
    map_area_100 = 0
    map_area_200 = 0
    map_area_300 = 0
    map_area_400 = 0
    map_area_500 = 0
    count_map_area_100 = 0
    count_map_area_200 = 0
    count_map_area_300 = 0
    count_map_area_400 = 0
    count_map_area_500 = 0

    # random
    log_start = time.time()
    last_decision = []
    total_usage = []

    pre_g_points = []

    target_point = []

    # Map
    agent_map_area = [] 
    all_agent_map_area = 0 

    # logging.info(f"num agents: {num_agents}")

    while count_episodes < num_episodes:
        observations = env.reset()
        # print(observations[0])
        for i in range(num_agents):
            agent[i].reset()
            if 'objectnav_hm3d' in args.task_config:
                agent_GT[i].reset()
        
        history_nodes.clear()
        history_score.clear()
        history_count.clear()
        history_states.clear()
        pre_g_points.clear()
        target_point.clear()

        goal_points.clear()
        agent_map_area.clear()
        all_agent_map_area = 0
        for j in range(num_agents):
            goal_points.append([0, 0])
            agent_map_area.append(0)

        is_labeled = False
        is_dif = False

        while not env.episode_over:
            
            all_rgb = [] 

            start = time.time()
            count_rotating = 0
            action = []
            
            for j in range(num_agents):
                action.append(0)
                
                
            full_map = []
            full_map1 = []
            visited_vis = []
            pose_pred = []
            agent_objs = {} 
            agent_FrontierList = [] 
            agent_TargetEdgeMap = []
            agent_TargetPointMap = []
            agent_MapPred = []

            for i in range(num_agents):
                agent[i].mapping(observations[i])
                if 'objectnav_hm3d' in args.task_config:
                    agent_GT[i].mapping(observations[i])
                # local_map1, _ = torch.max(agent[i].local_map.unsqueeze(0), 0)
                local_map1 = agent[i].local_map[4:, :,:].argmax(0)#[480*480]
                if 'objectnav_hm3d' in args.task_config:
                    element_15_count = torch.sum(local_map1 != 15).item()
                elif 'objectnav_mp3d' in args.task_config:
                    element_15_count = torch.sum(local_map1 != 20).item()
                # local_map_non_zero_count = torch.count_nonzero(local_map1)
                agent_map_area[i] = element_15_count 

                full_map.append(agent[i].local_map)
                visited_vis.append(agent[i].visited_vis)
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs

                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                pos = (
                    (start_x * 100. / args.map_resolution - gy1)
                    * 480 / agent[i].visited_vis.shape[0],
                    (agent[i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                    * 480 / agent[i].visited_vis.shape[1],
                    np.deg2rad(-start_o)
                )
                pose_pred.append(pos)

                    
            full_map2 = torch.cat([fm.unsqueeze(0) for fm in full_map], dim=0)


            full_map_pred, _ = torch.max(full_map2, 0)

            if 'objectnav_hm3d' in args.task_config:
                element_15_count = torch.sum(full_map_pred[4:, :,:].argmax(0) != 15).item()
            elif 'objectnav_mp3d' in args.task_config:
                element_15_count = torch.sum(full_map_pred[4:, :,:].argmax(0) != 20).item()
            all_agent_map_area = element_15_count 

            Wall_list, full_Frontier_list, full_target_edge_map, full_target_point_map = get_Frontiers(full_map_pred)


            if agent[0].l_step+1 == 100 and all_agent_map_area <= 160000:
                map_area_100 += all_agent_map_area
                count_map_area_100 += 1
            elif agent[0].l_step+1 == 200 and all_agent_map_area <= 160000:
                map_area_200 += all_agent_map_area
                count_map_area_200 += 1
            elif agent[0].l_step+1 == 300 and all_agent_map_area <= 160000:
                map_area_300 += all_agent_map_area
                count_map_area_300 += 1
            elif agent[0].l_step+1 == 400 and all_agent_map_area <= 160000:
                map_area_400 += all_agent_map_area
                count_map_area_400 += 1
            elif agent[0].l_step+1 == 500 and all_agent_map_area <= 160000:
                map_area_500 += all_agent_map_area
                count_map_area_500 += 1

            if agent[0].goal_id + 4 > 24:
                break

            if (agent[0].l_step % args.num_local_steps == args.num_local_steps - 1 or agent[0].l_step == 0) \
                and all(not agent[i].Find_Goal for i in range(num_agents)):
                
                for j in range(num_agents):
                    # goal_points[j] = [9999, 9999]
                    # agent[j].EXIT = False
                    agent[j].Perception_PR = 0
                    # agent[j].Max_Perception_Angle = 360
                    # agent[j].count_rerotation = 0
                
                agents_seg_list = Objects_Extract(args, full_map_pred, args.use_sam)

                pre_goal_points.clear()
                if len(cur_goal_points) > 0:
                    pre_goal_points = cur_goal_points.copy()
                    cur_goal_points.clear()
                    
                if len(full_target_point_map) > 0:
                    full_Frontiers_dict = {}
                    for j in range(len(full_target_point_map)):
                        full_Frontiers_dict['frontier_' + str(j)] = f"<centroid: {full_target_point_map[j][0], full_target_point_map[j][1]}, number: {full_Frontier_list[j]}>"
                    logging.info(f'=====> Frontier: {full_Frontiers_dict}')

                    if len(history_nodes) > 0:
                        logging.info(f'=====> history_nodes: {history_nodes}')
                        logging.info(f'=====> history_score: {history_score}')

                    # full_sem_map = Decision_Generation_Vis(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                    #                 agent[0].goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict, goal_points)
                        
                    # VLM_Decision_Prompt_Meta = form_prompt_for_DecisionVLM_MetaPreprocess()
                    # _, Decision_Pred_Meta = cogvlm2.simple_image_chat(User_Prompt=VLM_Decision_Prompt_Meta, 
                    #                                                                 return_string_probabilities=None, img=full_sem_map)
                    # Decision_Pred_Meta = '''
                    # Scenario exploration analysis module: Yes
                    # Scene object detection module: Yes
                    # Scenario exploration analysis module: Yes
                    # '''
                    ##### VLM Process :>
                    
                    for j in range(num_agents):
                        agent[j].is_Frontier = True
                        rgb = agent[j].rgb_vis
                        
                        # full_rgb1.append(full_rgb)
                        all_rgb.append(rgb)
                        goal_name = agent[j].goal_name
                        if args.yolo == 'yolov9':
                            agent_objs[f"agent_{j}"] = yolo.run(rgb) 
                        else:
                            yolo_output = yolo(source=rgb,conf=0.2)
                            yolo_mapping = [yolo_output[0].names[int(c)] for c in yolo_output[0].boxes.cls]
                            agent_objs[f"agent_{j}"] = {k: v for k, v in zip(yolo_mapping, yolo_output[0].boxes.conf)}
                        # logging.info(agent_objs)
                        
                        # agents_seg_list = Objects_Extract(local_map1, args.use_sam)
                        single_map = [full_map[j]]

                        full_map1.append(torch.cat([fm.unsqueeze(0) for fm in single_map], dim=0))
                        full_map_pred1, _ = torch.max(full_map1[j], 0)
                        Wall_list, Frontier_list, target_edge_map, target_point_map = get_Frontiers(full_map_pred1)
                        agent_FrontierList.append(Frontier_list)
                        agent_TargetEdgeMap.append(target_edge_map)
                        agent_TargetPointMap.append(target_point_map)
                        agent_MapPred.append(full_map_pred1)

                        

                        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[j].planner_pose_inputs
                        r, c = start_y, start_x
                        start = [int(r * 100.0 / args.map_resolution - gx1),
                                int(c * 100.0 / args.map_resolution - gy1)]
                        start = pu.threshold_poses(start, agent[j].local_map[0, :, :].cpu().numpy().shape)

                        # full_map_pred_copy = deepcopy(full_map_pred)
                        
                        if len(pre_goal_points) > 0:
                            # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                            #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=pre_goal_points[j])
                            sem_map, sem_map_frontier, sem_map_diffusion, Diffusion_Frontiers_dict, \
                                sem_map_labeled, sem_map_labeled_diffusion, diffusion_agents_seg_list, is_dif = \
                                Decision_Generation_Vis(args, sde, S_sde, model, agent[j], agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                                    agent[0].goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=pre_goal_points[j])
                        else:
                            # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                            #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=None)
                            sem_map, sem_map_frontier, sem_map_diffusion, Diffusion_Frontiers_dict, \
                                sem_map_labeled, sem_map_labeled_diffusion, diffusion_agents_seg_list, is_dif = \
                                Decision_Generation_Vis(args, sde, S_sde, model, agent[j], agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                                    agent[0].goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=None)
                        is_labeled = True
                        # full_rgb = np.hstack((rgb, sem_map))

                        #### 感知VLM
                        Caption_Prompt, VLM_Perception_Prompt = form_prompt_for_PerceptionVLM(goal_name, agent_objs[f'agent_{j}'], args.yolo)
                        _, Scene_Information = cogvlm2.simple_image_chat(User_Prompt=Caption_Prompt, 
                                                                        return_string_probabilities=None, img=rgb)
                        Perception_Rel, Perception_Pred = cogvlm2.COT2(User_Prompt1=Caption_Prompt, 
                                                                       User_Prompt2=VLM_Perception_Prompt,
                                                                       cot_pred1=Scene_Information,
                                                                       return_string_probabilities="[Yes, No]", img=rgb)
                        Perception_Rel = np.array(Perception_Rel)
                        Perception_PR = Perception_weight_decision(Perception_Rel, Perception_Pred)
                        logging.info(f"Agent_{j}--VLM_PerceptionPR: {Perception_PR}")
                        # agents_VLM_Rel[f"Agent_{i}--VLM_PerceptionRel"] = Perception_Rel
                        # agents_VLM_Pred[f"Agent_{i}--VLM_PerceptionPred"] = Perception_Pred
                        # agents_VLM_PR[f"Agent_{i}--VLM_PerceptionPR"] = Perception_PR

                        

                        is_exist_oldhistory = False
                        if len(history_nodes) > 0:
                            closest_index = -1
                            min_distance = float('inf')
                            new_x, new_y = start
                            for i, (x, y) in enumerate(history_nodes):
                                distance = math.sqrt((x - new_x) * (x - new_x) + (y - new_y) * (y - new_y))
                                if distance < 25 and distance < min_distance:
                                    min_distance = distance
                                    closest_index = i
                                    is_exist_oldhistory = True

                            if  is_exist_oldhistory == False:
                                history_nodes.append(start)
                                history_count.append(1)
                                history_state = np.zeros(360)
                            else:
                                history_count[closest_index] = history_count[closest_index] + 1

                            
                        else:
                            history_nodes.append(start)
                            history_count.append(1)
                            history_state = np.zeros(360)

                        
                        cur_goal_points.append(start)

                        if len(agent_TargetPointMap[j]) > 0:
                            
                            # Frontiers_dict = {}
                            # for k in range(len(agent_TargetPointMap[j])):
                            #     Frontiers_dict['frontier_' + str(k)] = f"<centroid: {agent_TargetPointMap[j][k][0], agent_TargetPointMap[j][k][1]}, number: {agent_FrontierList[j][k]}>"
                            # Agent States
                            
                            logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')

                            if is_dif:
                                if len(Diffusion_Frontiers_dict) > 0:
                                    full_Frontiers_dict = Diffusion_Frontiers_dict
                            if len(history_nodes) > 0:
                                
                                if is_dif:
                                    if len(pre_goal_points) > 0:
                                        FN_Prompt = form_prompt_for_FN(goal_name, agents_seg_list, Perception_PR, pre_goal_points[j], Diffusion_Frontiers_dict, start, history_nodes)
                                    else:
                                        FN_Prompt = form_prompt_for_FN(goal_name, agents_seg_list, Perception_PR, pre_goal_points, Diffusion_Frontiers_dict, start, history_nodes)
                                    FN_Rel, FN_Decision = cogvlm2.simple_image_chat(User_Prompt=FN_Prompt, 
                                                                                            return_string_probabilities="[Yes, No]", img=sem_map_diffusion)
                                else:
                                    if len(pre_goal_points) > 0:
                                        FN_Prompt = form_prompt_for_FN(goal_name, agents_seg_list, Perception_PR, pre_goal_points[j], full_Frontiers_dict, start, history_nodes)
                                    else:
                                        FN_Prompt = form_prompt_for_FN(goal_name, agents_seg_list, Perception_PR, pre_goal_points, full_Frontiers_dict, start, history_nodes)
                                    FN_Rel, FN_Decision = cogvlm2.simple_image_chat(User_Prompt=FN_Prompt, 
                                                                                            return_string_probabilities="[Yes, No]", img=sem_map)

                                FN_PR = Perception_weight_decision(FN_Rel, FN_Decision)
                                logging.info(f"Agent_{j}--FN_PR: {FN_PR}")
                                if FN_PR == 'Neither':
                                    FN_PR = FN_Rel

                                
                                
                                angle_score = Perception_PR[0] * 2 + FN_PR[0]
                                agent[j].angle_score = angle_score
                                c_angle = int(start_o % 360)

                                if is_exist_oldhistory == False:
                                    if c_angle >= 39 and c_angle < 321:
                                        history_state[c_angle-39:c_angle+39] = angle_score
                                    elif c_angle < 39:
                                        history_state[:c_angle+39] = angle_score
                                        history_state[360-c_angle-39:] = angle_score

                                    elif c_angle >= 321:
                                        history_state[c_angle-39:] = angle_score
                                        history_state[:c_angle+39-360] = angle_score
                                    h_score = history_state.sum()
                                    history_states.append(history_state)
                                    history_score.append(h_score)
                                else:
                                    if c_angle >= 39 and c_angle < 321:
                                        history_states[closest_index][c_angle-39:c_angle+39] = angle_score
                                    elif c_angle < 39:
                                        history_states[closest_index][:c_angle] = angle_score
                                        history_states[closest_index][360-c_angle:] = angle_score
                                    elif c_angle >= 321:
                                        history_states[closest_index][c_angle:] = angle_score
                                        history_states[closest_index][:360-c_angle] = angle_score
                                    h_score = history_states[closest_index].sum() / history_count[closest_index]
                                    history_score[closest_index] = h_score

                            logging.info(f'=====> history_nodes: {history_nodes}')
                            logging.info(f'=====> history_score: {history_score}')
                            # Scores = []
                            if j == 0:
                                history_nodes_copy = history_nodes.copy()
                                history_score_copy = history_score.copy()
                                full_Frontiers_dict_copy = full_Frontiers_dict.copy()
                            else:
                                missing_key_F = []
                                if len(full_Frontiers_dict) == 4:
                                    frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                                elif len(full_Frontiers_dict) == 3:
                                    frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2']
                                elif len(full_Frontiers_dict) == 2:
                                    frontier_keys = ['frontier_0', 'frontier_1']
                                else:
                                    frontier_keys = ['frontier_0']

                                for element in full_Frontiers_dict.keys():
                                    if element not in full_Frontiers_dict_copy.keys():
                                        missing_key_F.append(element)
                                # for element in history_nodes:
                                #     if element not in history_nodes_copy:
                                #         missing_index_H.append(element.index(element))
                            if FN_PR[0] >= 0.2 or agent[j].l_step <= 200:
                                if is_dif:
                                    seg_list = agents_seg_list
                                else:
                                    seg_list = agents_seg_list
                                Meta_Information_Prompt = form_prompt_for_DecisionVLM_Frontier_COT1()
                                if len(pre_goal_points) > 0:
                                    Meta_Prompt = form_prompt_for_DecisionVLM_Frontier_COT2(Scene_Information, seg_list, pre_goal_points[j], goal_name, start, full_Frontiers_dict_copy)
                                else:
                                    Meta_Prompt = form_prompt_for_DecisionVLM_Frontier_COT2(Scene_Information, seg_list, pre_goal_points, goal_name, start, full_Frontiers_dict_copy)
                                
                                if is_dif:
                                    _, Map_Information = cogvlm2.simple_image_chat(User_Prompt=Meta_Information_Prompt, 
                                                                            return_string_probabilities=None, img=sem_map_diffusion)
                                    Meta_Score, Meta_Choice = cogvlm2.COT2(User_Prompt1=Meta_Information_Prompt, 
                                                                                User_Prompt2=Meta_Prompt,
                                                                                cot_pred1=Map_Information,
                                                                                return_string_probabilities="[A, B, C, D]", img=sem_map_diffusion)
                                else:
                                    _, Map_Information = cogvlm2.simple_image_chat(User_Prompt=Meta_Information_Prompt, 
                                                                            return_string_probabilities=None, img=sem_map)
                                    Meta_Score, Meta_Choice = cogvlm2.COT2(User_Prompt1=Meta_Information_Prompt, 
                                                                                User_Prompt2=Meta_Prompt,
                                                                                cot_pred1=Map_Information,
                                                                                return_string_probabilities="[A, B, C, D]", img=sem_map)

                                Final_PR = Perception_weight_decision4(Meta_Score, Meta_Choice)
                                
                            else:
                                # 由于不稳定性，将其替换为分数最高的nodes
                                # Meta_Prompt = form_prompt_for_DecisionVLM_History(pre_goal_points[j], goal_name, start, history_score_copy, history_nodes_copy)
                                # Meta_Score, Meta_Choice = cogvlm2.COT3(User_Prompt1=VLM_Perception_Prompt, 
                                #                             User_Prompt2=FN_Prompt,
                                                            # User_Prompt3=Meta_Prompt,
                                                            # cot_pred1=Perception_Pred,
                                                            # cot_pred2=FN_Decision,
                                                            # return_string_probabilities="[a, b, c, d]", img=full_rgb)
                                # Decisions.append(Meta_Choice)
                                # Final_PR = Perception_weight_decision26(Meta_Score, Meta_Choice)
                                Final_PR = history_score_copy

                            logging.info(f"Agent_{j}--Decision_PR: {Final_PR}")

                            # Scores.append(Final_PR)
                            Choice = Final_PR.index(max(Final_PR))
                            
                            
                            if FN_PR[0] >= 0.2 or agent[j].l_step <= 200:
                                logging.info(f"VLM Choice: Agent_{j}-frontier_{Choice}")
                                Choice2 = Meta_Score.index(max(Meta_Score))

                                if len(full_Frontiers_dict) == 1:
                                    goal_points[j] = [int(x) for x in full_Frontiers_dict['frontier_0'].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                
                                elif len(full_Frontiers_dict) == 2 and num_agents == 3:
                                    if j == 0:
                                        for i, key in enumerate(frontier_keys):
                                            if Choice == i:
                                                if key in full_Frontiers_dict_copy:
                                                    goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                                    del full_Frontiers_dict_copy[key]
                                    elif j == 1:
                                        if len(missing_key_F) != 0:
                                            for keys in missing_key_F:
                                                frontier_keys.remove(keys)
                                        for i, key in enumerate(frontier_keys):
                                            goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                    else:
                                        if len(missing_key_F) != 0:
                                            for keys in missing_key_F:
                                                frontier_keys.remove(keys)
                                        for i, key in enumerate(frontier_keys):
                                            goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                
                                
                                else:
                                    if j > 0:
                                        if len(missing_key_F) != 0:
                                            for keys in missing_key_F:
                                                frontier_keys.remove(keys)
                                    else:
                                        if len(full_Frontiers_dict) == 4:
                                            frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                                        elif len(full_Frontiers_dict) == 3:
                                            frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2']
                                        elif len(full_Frontiers_dict) == 2:
                                            frontier_keys = ['frontier_0', 'frontier_1']
                                        else:
                                            frontier_keys = ['frontier_0']

                                    invalid_answer = False
                                    for i, key in enumerate(frontier_keys):
                                        if Choice == i:
                                            if key in full_Frontiers_dict_copy:
                                                goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                                del full_Frontiers_dict_copy[key]
                                            else:
                                                invalid_answer = True
                                            break
                                    if invalid_answer:
                                        for i, key in enumerate(frontier_keys):
                                            if Choice2 == i:
                                                try:
                                                    goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                                    del full_Frontiers_dict_copy[key]
                                                    break
                                                except:
                                                    goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[frontier_keys[0]].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                                    del full_Frontiers_dict_copy[frontier_keys[0]]
                                                    break
                                        

                            else:
                                logging.info(f"VLM Choice: Agent_{j}-history_{Choice}")
                                if len(history_nodes_copy)==1:
                                    goal_points[j] = history_nodes_copy[0]
                                else:
                                    for i in range(len(history_nodes_copy)):
                                        if Choice == i:
                                            goal_points[j] = history_nodes_copy[i]
                                            del history_nodes_copy[i]
                                            del history_score_copy[i]
                                            break

                            
                            
                        else:
                            logging.info(f'===== Agent_{j} No Frontier, Random Mode =====')
                            agent[j].is_Frontier = False
                            c_angle = int(start_o % 360)
                            angle_score = Perception_PR[0] * 2
                            agent[j].angle_score = angle_score

                            if is_exist_oldhistory == False:
                                if c_angle >= 39 and c_angle < 321:
                                    history_state[c_angle-39:c_angle+39] = angle_score
                                elif c_angle < 39:
                                    history_state[:c_angle+39] = angle_score
                                    history_state[360-c_angle-39:] = angle_score

                                elif c_angle >= 321:
                                    history_state[c_angle-39:] = angle_score
                                    history_state[:c_angle+39-360] = angle_score
                                h_score = history_state.sum()
                                history_states.append(history_state)
                                history_score.append(h_score)
                            else:
                                if c_angle >= 39 and c_angle < 321:
                                    history_states[closest_index][c_angle-39:c_angle+39] = angle_score
                                elif c_angle < 39:
                                    history_states[closest_index][:c_angle] = angle_score
                                    history_states[closest_index][360-c_angle:] = angle_score
                                elif c_angle >= 321:
                                    history_states[closest_index][c_angle:] = angle_score
                                    history_states[closest_index][:360-c_angle] = angle_score
                                h_score = history_states[closest_index].sum() / history_count[closest_index]
                                history_score[closest_index] = h_score

                            if j == 0:
                                history_nodes_copy = history_nodes.copy()
                                history_score_copy = history_score.copy()
                                full_Frontiers_dict_copy = full_Frontiers_dict.copy()
                            
                            if len(full_Frontiers_dict) == 1:
                                logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')
                                actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                                goal_points[j] = [int(actions[0]), int(actions[1])]
                            else:
                                if  j == 0:
                                    frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                                logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')
                                actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                                goal_points[j] = [int(actions[0]), int(actions[1])]
                            

                else:
                    
                    logging.info(f'===== No Frontier, Random Mode===== ')
                    logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')
                    
                    for j in range(num_agents):
                        agent[j].is_Frontier = False
                        # rgb = observations[j]['rgb'].astype(np.uint8)
                        rgb = agent[j].rgb_vis
                        
                        # full_rgb1.append(full_rgb)
                        all_rgb.append(rgb)
                        goal_name = agent[j].goal_name
                        if args.yolo == 'yolov9':
                            agent_objs[f"agent_{j}"] = yolo.run(rgb) 
                        else:
                            yolo_output = yolo(source=rgb,conf=0.2)
                            yolo_mapping = [yolo_output[0].names[int(c)] for c in yolo_output[0].boxes.cls]
                            agent_objs[f"agent_{j}"] = {k: v for k, v in zip(yolo_mapping, yolo_output[0].boxes.conf)}
                        # logging.info(agent_objs)

                        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[j].planner_pose_inputs
                        r, c = start_y, start_x
                        start = [int(r * 100.0 / args.map_resolution - gx1),
                                int(c * 100.0 / args.map_resolution - gy1)]
                        start = pu.threshold_poses(start, agent[j].local_map[0, :, :].cpu().numpy().shape)
                        
                        cur_goal_points.append(start)

                        #### 感知VLM
                        Caption_Prompt, VLM_Perception_Prompt = form_prompt_for_PerceptionVLM(goal_name, agent_objs[f'agent_{j}'], args.yolo)
                        _, Scene_Information = cogvlm2.simple_image_chat(User_Prompt=Caption_Prompt, 
                                                                        return_string_probabilities=None, img=rgb)
                        Perception_Rel, Perception_Pred = cogvlm2.COT2(User_Prompt1=Caption_Prompt, 
                                                                       User_Prompt2=VLM_Perception_Prompt,
                                                                       cot_pred1=Scene_Information,
                                                                       return_string_probabilities="[Yes, No]", img=rgb)
                        Perception_Rel = np.array(Perception_Rel)
                        Perception_PR = Perception_weight_decision(Perception_Rel, Perception_Pred)
                        logging.info(f"Agent_{j}--VLM_PerceptionPR: {Perception_PR}")

                        is_exist_oldhistory = False
                        if len(history_nodes) > 0:
                            closest_index = -1
                            min_distance = float('inf')
                            new_x, new_y = start
                            for i, (x, y) in enumerate(history_nodes):
                                distance = math.sqrt((x - new_x) * (x - new_x) + (y - new_y) * (y - new_y))
                                if distance < 25 and distance < min_distance:
                                    min_distance = distance
                                    closest_index = i
                                    is_exist_oldhistory = True

                            if  is_exist_oldhistory == False:
                                history_nodes.append(start)
                                history_count.append(1)
                                history_state = np.zeros(360)
                            else:
                                history_count[closest_index] = history_count[closest_index] + 1

                            
                        else:
                            history_nodes.append(start)
                            history_count.append(1)
                            history_state = np.zeros(360)


                        angle_score = Perception_PR[0] * 2
                        agent[j].angle_score = angle_score
                        c_angle = int(start_o % 360)

                        if is_exist_oldhistory == False:
                            if c_angle >= 39 and c_angle < 321:
                                history_state[c_angle-39:c_angle+39] = angle_score
                            elif c_angle < 39:
                                history_state[:c_angle+39] = angle_score
                                history_state[360-c_angle-39:] = angle_score

                            elif c_angle >= 321:
                                history_state[c_angle-39:] = angle_score
                                history_state[:c_angle+39-360] = angle_score
                            h_score = history_state.sum()
                            history_states.append(history_state)
                            history_score.append(h_score)
                        else:
                            if c_angle >= 39 and c_angle < 321:
                                history_states[closest_index][c_angle-39:c_angle+39] = angle_score
                            elif c_angle < 39:
                                history_states[closest_index][:c_angle] = angle_score
                                history_states[closest_index][360-c_angle:] = angle_score
                            elif c_angle >= 321:
                                history_states[closest_index][c_angle:] = angle_score
                                history_states[closest_index][:360-c_angle] = angle_score
                            h_score = history_states[closest_index].sum() / history_count[closest_index]
                            history_score[closest_index] = h_score


                        actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                        goal_points[j] = [int(actions[0]), int(actions[1])]

                for i in range(num_agents):
                    if len(pre_g_points) == 0:
                        break
                    if calculate_distance(cur_goal_points[i], pre_g_points[i]) >= 25 and agent[i].is_Frontier == True:
                        # print(calculate_distance(cur_goal_points[i], pre_g_points[i]))
                        goal_points[i] = pre_g_points[i]

                for i in range(num_agents):
                    if len(pre_goal_points) > 0 and calculate_distance(pre_goal_points[i], cur_goal_points[i]) <= 2.5:
                        actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                        goal_points[i] = [int(actions[0]), int(actions[1])]
                
      
                
                logging.info(f"goal_points: {goal_points}")
                pre_g_points = goal_points.copy()
                logging.info("===== Starting local strategy ===== ")
            
            

            for i in range(num_agents):
                if len(target_point) > 0:
                    for j in range(num_agents):
                        goal_points[j] = target_point
                action[i] = agent[i].act(goal_points[i])
                if 'objectnav_hm3d' in args.task_config:
                    _ = agent_GT[i].act(goal_points[i])
                if action[i] == 0:
                    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                    r, c = start_y, start_x
                    start = [int(r * 100.0 / args.map_resolution - gx1),
                            int(c * 100.0 / args.map_resolution - gy1)]
                    start = pu.threshold_poses(start, agent[i].local_map[0, :, :].cpu().numpy().shape)
                    target_point = start.copy()
            # logging.info(f"actions: {action}")
            observations = env.step(action)
            
            # exit(0)
                    
            
                        
            # if count_rotating == 2:
            #     exit(0)
            # ------------------------------------------------------------------

                # Debug
            
            if args.visualize or args.print_images: 
                if num_agents == 1:
                    vis_ep_dir = '{}/episodes/eps_{}/Agent0_vis'.format(
                        dump_dir, agent[0].episode_n)
                    if not os.path.exists(vis_ep_dir):
                        os.makedirs(vis_ep_dir)
                    # Legend = cv2.imread("img/legend.png")
                    # height, _ = sem_map.shape[:2]
                    # legend_resized = cv2.resize(Legend, (Legend.shape[1], height))
                    # img_show = np.hstack((sem_map, legend_resized))
                    # img_show = observations[0]['rgb'].astype(np.uint8)
                    # img_show2 = observations[1]['rgb'].astype(np.uint8)
                    img_show = agent[0].rgb_vis
                    fn = '{}/episodes/eps_{}/Agent0_vis/VisStep-{}.png'.format(
                        dump_dir, agent[0].episode_n,
                        agent[0].l_step)
                    cv2.imwrite(fn, img_show)
                    # cv2.imwrite(fn2, img_show2)    

                    if is_labeled:
                        Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                                agent[0].goal_id, visited_vis, full_target_edge_map, Frontiers_dict=None, goal_points=goal_points,
                                is_labeled=is_labeled, is_dif=is_dif, sem_map_labeled=sem_map_labeled, sem_map_diffusion=sem_map_labeled_diffusion)
                    else:
                        Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                                agent[0].goal_id, visited_vis, full_target_edge_map, Frontiers_dict=None, goal_points=goal_points,
                                is_labeled=is_labeled, is_dif=is_dif, sem_map_labeled=None, sem_map_diffusion=None)
                        
                if num_agents == 2:
                    vis_ep_dir = '{}/episodes/eps_{}/Agent0_vis'.format(
                        dump_dir, agent[0].episode_n)
                    vis_ep_dir2 = '{}/episodes/eps_{}/Agent1_vis'.format(
                        dump_dir, agent[0].episode_n)
                    if not os.path.exists(vis_ep_dir):
                        os.makedirs(vis_ep_dir)
                    if not os.path.exists(vis_ep_dir2):
                        os.makedirs(vis_ep_dir2)
                    # Legend = cv2.imread("img/legend.png")
                    # height, _ = sem_map.shape[:2]
                    # legend_resized = cv2.resize(Legend, (Legend.shape[1], height))
                    # img_show = np.hstack((sem_map, legend_resized))
                    # img_show = observations[0]['rgb'].astype(np.uint8)
                    # img_show2 = observations[1]['rgb'].astype(np.uint8)
                    img_show = agent[0].rgb_vis
                    img_show2 = agent[1].rgb_vis
                    fn = '{}/episodes/eps_{}/Agent0_vis/VisStep-{}.png'.format(
                        dump_dir, agent[0].episode_n,
                        agent[0].l_step)
                    fn2 = '{}/episodes/eps_{}/Agent1_vis/VisStep-{}.png'.format(
                        dump_dir, agent[0].episode_n,
                        agent[0].l_step)
                    # print(fn)
                    cv2.imwrite(fn, img_show)
                    cv2.imwrite(fn2, img_show2)    
                    if is_labeled:
                        Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                                agent[0].goal_id, visited_vis, full_target_edge_map, Frontiers_dict=None, goal_points=goal_points,
                                is_labeled=is_labeled, is_dif=is_dif, sem_map_labeled=sem_map_labeled, sem_map_diffusion=sem_map_labeled_diffusion)
                    else:
                        Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                                agent[0].goal_id, visited_vis, full_target_edge_map, Frontiers_dict=None, goal_points=goal_points,
                                is_labeled=is_labeled, is_dif=is_dif, sem_map_labeled=None, sem_map_diffusion=None)

            # logging.info(f"full_map_pred.shape: {full_map_pred.shape}") # [20,480,480] HM-3D

##############################################===Metric===##############################################

        count_episodes += 1
        # obj_SR['num_'+agent[0].goal_name] += 1
        count_step += agent[0].l_step
        if agent[0].l_step > 160:
            count_DF += 1

        # ------------------------------------------------------------------
        ##### Logging
        # ------------------------------------------------------------------
        if is_labeled:
            del sem_map_labeled
        if is_dif:
            del sem_map_labeled_diffusion

        if args.not_explore == 1:
            log_end = time.time()
            time_elapsed = time.gmtime(log_end - log_start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(count_step),
                "average timesteps {},".format(count_step / count_episodes),
            ]) + '\n'

            if agent[0].goal_id + 4 > 24:
                log += '==========Unknown Label=========='
                log += '\n'
                for k, v in agg_metrics.items():
                    if k == 'multi_Total_SR':
                        for i in range(num_agents):
                            if 'objectnav_hm3d' in args.task_config:
                                if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                                    agg_metrics[k] += 1
                                    if agent[i].l_step > 160:
                                        count_DFSR += 1
                                    if agg_metrics[k] > count_episodes:
                                        agg_metrics[k] = count_episodes
                                    agg_metrics['multi_Navigation_SR'] += 1
                                    if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                        agg_metrics['multi_Navigation_SR'] = count_episodes
                                    break
                                elif agent[i].Find_Goal and agent_GT[i].Find_Goal == False:
                                    agg_metrics[k] += 0
                                    agg_metrics['multi_Navigation_SR'] += 1
                                    if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                        agg_metrics['multi_Navigation_SR'] = count_episodes
                                    break
                            else:
                                if agent[i].Find_Goal:
                                    agg_metrics[k] += 1
                                    if agent[i].l_step > 160:
                                        count_DFSR += 1
                                    if agg_metrics[k] > count_episodes:
                                        agg_metrics[k] = count_episodes
                                    agg_metrics['multi_Navigation_SR'] += 1
                                    if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                        agg_metrics['multi_Navigation_SR'] = count_episodes
                                    break

                spls = []
                for i in range(num_agents):
                    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                    r, c = start_y, start_x
                    start = [int(r * 100.0 / args.map_resolution - gx1),
                    int(c * 100.0 / args.map_resolution - gy1)]
                    start = pu.threshold_poses(start, agent[i].local_map[0, :, :].cpu().numpy().shape)
                    if 'objectnav_hm3d' in args.task_config:
                        if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                            spl = agent[i].get_spl(success=1,cur_loc=start)
                        else:
                            spl = agent[i].get_spl(success=0,cur_loc=start)
                    else:
                        if agent[i].Find_Goal:
                            spl = agent[i].get_spl(success=1,cur_loc=start)
                        else:
                            spl = agent[i].get_spl(success=0,cur_loc=start)
                    agg_metrics['multi_SPL'][f'Agent_{i}'] = spl
                    agg_metrics['multi_SoftSPL'][f'Agent_{i}'] += spl
                    spls.append(spl)
                agg_metrics['SPL'] = max(spls)
                agg_metrics['SoftSPL'] += max(spls)
                for agent_name, SPL in agg_metrics['multi_SPL'].items():
                    SoftSPL = agg_metrics['multi_SoftSPL'][agent_name] / count_episodes
                    log += f"{agent_name}" + "---SPL: {:.3f}, SoftSPL: {:.3f}".format(SPL, SoftSPL)
                    log += '\n'
                
                if count_map_area_100 != 0:
                    log += f"Explore area 100" + "---Area: {:.3f}".format(map_area_100 / count_map_area_100)
                    log += '\n'
                else:
                    log += "Explore area 100---Area: 0.000"
                    log += '\n'
                if count_map_area_200 != 0:
                    log += f"Explore area 200" + "---Area: {:.3f}".format(map_area_200 / count_map_area_200)
                    log += '\n'
                else:
                    log += "Explore area 200---Area: 0.000"
                    log += '\n'
                if count_map_area_300 != 0:
                    log += f"Explore area 300" + "---Area: {:.3f}".format(map_area_300 / count_map_area_300)
                    log += '\n'
                else:
                    log += "Explore area 300---Area: 0.000"
                    log += '\n'
                if count_map_area_400 != 0:
                    log += f"Explore area 400" + "---Area: {:.3f}".format(map_area_400 / count_map_area_400)
                    log += '\n'
                else:
                    log += "Explore area 400---Area: 0.000"
                    log += '\n'
                if count_map_area_500 != 0:
                    log += f"Explore area 500" + "---Area: {:.3f}".format(map_area_500 / count_map_area_500)
                    log += '\n'
                else:
                    log += "Explore area 500---Area: 0.000"
                    log += '\n'

                log += "multi_Total_SR: {:.3f}, ".format(agg_metrics['multi_Total_SR'] / count_episodes)
                log += "multi_Navigation_SR/SR: {:.0f}/{:.0f}, ".format(agg_metrics['multi_Navigation_SR'], agg_metrics['multi_Total_SR'])
                log += "multi_SPL: {:.3f}, ".format(agg_metrics['SPL'])
                log += "multi_SoftSPL: {:.3f}, ".format(agg_metrics['SoftSPL'] / count_episodes)
                log += "DF: {:.3f}, DFSR: {:.3f} ".format(count_DF / count_episodes, count_DFSR / (count_DF+1e-3))
                log += " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)
            else:
                # metrics = env.get_metrics()

                for k, v in agg_metrics.items():
                    if k == 'multi_Total_SR':
                        for i in range(num_agents):
                            if 'objectnav_hm3d' in args.task_config:
                                if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                                    agg_metrics[k] += 1
                                    if agent[i].l_step > 160:
                                        count_DFSR += 1
                                    if agg_metrics[k] > count_episodes:
                                        agg_metrics[k] = count_episodes
                                    agg_metrics['multi_Navigation_SR'] += 1
                                    if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                        agg_metrics['multi_Navigation_SR'] = count_episodes
                                    break
                                elif agent[i].Find_Goal and agent_GT[i].Find_Goal == False:
                                    agg_metrics[k] += 0
                                    agg_metrics['multi_Navigation_SR'] += 1
                                    if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                        agg_metrics['multi_Navigation_SR'] = count_episodes
                                    break
                            else:
                                if agent[i].Find_Goal:
                                    agg_metrics[k] += 1
                                    if agent[i].l_step > 160:
                                        count_DFSR += 1
                                    if agg_metrics[k] > count_episodes:
                                        agg_metrics[k] = count_episodes
                                    agg_metrics['multi_Navigation_SR'] += 1
                                    if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                        agg_metrics['multi_Navigation_SR'] = count_episodes
                                    break
                spls = []
                for i in range(num_agents):
                    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                    r, c = start_y, start_x
                    start = [int(r * 100.0 / args.map_resolution - gx1),
                    int(c * 100.0 / args.map_resolution - gy1)]
                    start = pu.threshold_poses(start, agent[i].local_map[0, :, :].cpu().numpy().shape)
                    if 'objectnav_hm3d' in args.task_config:
                        if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                            spl = agent[i].get_spl(success=1,cur_loc=start)
                        else:
                            spl = agent[i].get_spl(success=0,cur_loc=start)
                    else:
                        if agent[i].Find_Goal:
                            spl = agent[i].get_spl(success=1,cur_loc=start)
                        else:
                            spl = agent[i].get_spl(success=0,cur_loc=start)
                    agg_metrics['multi_SPL'][f'Agent_{i}'] = spl
                    agg_metrics['multi_SoftSPL'][f'Agent_{i}'] += spl
                    spls.append(spl)
                agg_metrics['SPL'] = max(spls)
                agg_metrics['SoftSPL'] += max(spls)
                for agent_name, SPL in agg_metrics['multi_SPL'].items():
                    SoftSPL = agg_metrics['multi_SoftSPL'][agent_name] / count_episodes
                    log += f"{agent_name}" + "---SPL: {:.3f}, SoftSPL: {:.3f}".format(SPL, SoftSPL)
                    log += '\n'
                
                if count_map_area_100 != 0:
                    log += f"Explore area 100" + "---Area: {:.3f}".format(map_area_100 / count_map_area_100)
                    log += '\n'
                else:
                    log += "Explore area 100---Area: 0.000"
                    log += '\n'
                if count_map_area_200 != 0:
                    log += f"Explore area 200" + "---Area: {:.3f}".format(map_area_200 / count_map_area_200)
                    log += '\n'
                else:
                    log += "Explore area 200---Area: 0.000"
                    log += '\n'
                if count_map_area_300 != 0:
                    log += f"Explore area 300" + "---Area: {:.3f}".format(map_area_300 / count_map_area_300)
                    log += '\n'
                else:
                    log += "Explore area 300---Area: 0.000"
                    log += '\n'
                if count_map_area_400 != 0:
                    log += f"Explore area 400" + "---Area: {:.3f}".format(map_area_400 / count_map_area_400)
                    log += '\n'
                else:
                    log += "Explore area 400---Area: 0.000"
                    log += '\n'
                if count_map_area_500 != 0:
                    log += f"Explore area 500" + "---Area: {:.3f}".format(map_area_500 / count_map_area_500)
                    log += '\n'
                else:
                    log += "Explore area 500---Area: 0.000"
                    log += '\n'

                log += "multi_Total_SR: {:.3f}, ".format(agg_metrics['multi_Total_SR'] / count_episodes)
                log += "multi_Navigation_SR/SR: {:.0f}/{:.0f}, ".format(agg_metrics['multi_Navigation_SR'], agg_metrics['multi_Total_SR'])
                log += "multi_SPL: {:.3f}, ".format(agg_metrics['SPL'])
                log += "multi_SoftSPL: {:.3f} ".format(agg_metrics['SoftSPL'] / count_episodes)
                log += "DF: {:.3f}, DFSR: {:.3f} ".format(count_DF / count_episodes, count_DFSR / (count_DF+1e-3))
                log += " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)
            # log += "Total usage: " + str(sum(total_usage)) + ", average usage: " + str(np.mean(total_usage))
            log += '\n===================================='
            print(log)
            logging.info(log)
            fn = '{}/TEST.log'.format(log_dir)
            if count_episodes == 1:
                with open(fn,'w', encoding='utf-8') as f:
                    f.write(log)
                    f.write('\n')
            else:
                with open(fn,'a', encoding='utf-8') as f:
                    f.write(log)
                    f.write('\n')
        # ------------------------------------------------------------------


    # avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    # return avg_metrics
    

if __name__ == "__main__":
    main()