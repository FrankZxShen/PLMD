#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import re
import random
from typing import Dict, Optional
import math
import time
import logging

import numba
import numpy as np
import torch
import torch.nn as nn
import gym
from torchvision import transforms
import torch.nn.functional as F

from habitat.core.agent import Agent
from habitat.core.simulator import Observations

from matplotlib import pyplot as plt
from skimage import measure
import skimage.morphology
from PIL import Image
from copy import deepcopy
import numpy as np

import ContourDiffusion.utils as util

from utils.hdbscan_utils import HdbscanCluster
from skimage.color import gray2rgb, rgb2gray
from skimage.feature import canny
from constants import color_palette, object_category
from scipy.spatial import KDTree

import cv2

from semantic_mapping import Semantic_Mapping
import utils.pose as pu
from utils.fmm_planner import FMMPlanner
from utils.semantic_prediction import SemanticPredMaskRCNN
import utils.visualization as vu
from utils.visualization import save_legend
from arguments import get_args

from constants import (
    coco_categories, coco_categories_hm3d2mp3d,
    gibson_coco_categories, color_palette, category_to_id, object_category
)
from RedNet.RedNet_model import load_rednet
from constants import mp_categories_mapping, mp_categories_mapping21

from Grounded_SAM.grounded_sam_demo import vis_semantics
from Grounded_SAM.gsam import GSAM, convert_SAM
from Grounded_SAM.grounded_sam_demo import load_model, get_grounding_output, save_mask_data, show_mask, show_box

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

class LLM_Agent(Agent):
    def __init__(self, args, config_env, agent_id, device, model=None, sde=None, S_sde=None,\
                 model2=None, sde2=None, S_sde2=None) -> None:
        self.args = args
        self.agent_id = agent_id
        print("args: ", args)

        self.t = 0 ###TEST

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.object_category = deepcopy(object_category)

        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        self.device = args.device = device

        # ------------------------------------------------------------------
        ##### Semantic detecttion init SAM
        # ------------------------------------------------------------------
        if self.args.use_sam:
            use_ram = args.tag_freq > 0
            while True:
                try:
                    self.GSAM = GSAM(self.object_category[:-1], text_threshold=args.text_threshold, device=self.device, use_ram=use_ram)
                    break
                except Exception as ex:
                    logging.info(f"[ERROR]: {ex}, sleep for 20s...")
                    time.sleep(20)
                    continue
        
        # ------------------------------------------------------------------
        ##### Semantic detecttion init RedNet
        # ------------------------------------------------------------------
        else:
            self.sem_pred = SemanticPredMaskRCNN(args)
            self.red_sem_pred = load_rednet(
                self.device, ckpt='RedNet/model/rednet_semmap_mp3d_40.pth', 
                resize=True, # since we train on half-vision
            )
            self.red_sem_pred.eval()
            self.red_sem_pred.to(self.device)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        ##### initializations for planning:
        # ------------------------------------------------------------------
        self.selem = skimage.morphology.disk(3)

        self.last_sim_location = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.last_action = None
        self.col_width = None
        self.l_step = 0
        self.episode_n = 0
        self.collision_n = 0

        self.collision_s = 0
        self.replan_count = 0

        # ------------------------------------------------------------------
        ##### Diffusion init
        # ------------------------------------------------------------------
        self.model = model
        self.sde = sde
        self.S_sde=  S_sde
        self.model2 = model2
        self.sde2 = sde2
        self.S_sde2=  S_sde2

        # ------------------------------------------------------------------
        self.last_planning_window = None
        self.last_sem_map_vis = None
        self.count_windows = 0
        self.count_masks = 0
        self.count_full_masks = 0
        self.pre_g_points = None
        self.diffusion_output_local = None
        self.diffusion_output_global = None
        self.new_long_term_goal_point = None
        self.cluster = HdbscanCluster()
        self.new_long_term_goal = None
        self.new_pred_goal_map = None
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        ##### Visualization init
        # ------------------------------------------------------------------
        if args.visualize or args.print_images:
            self.vis_image = None
            self.rgb_vis = None
            self.set_legend()

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # ------------------------------------------------------------------
        

        # ------------------------------------------------------------------
        ##### Initialize map variables:
        ##### Full map consists of multiple channels containing the following:
        ##### 1. Obstacle Map
        ##### 2. Exploread Area
        ##### 3. Current Agent Location
        ##### 4. Past Agent Locations
        ##### 5,6,7,.. : Semantic Categories
        # ------------------------------------------------------------------
        nc = args.num_sem_categories + 4  # num channels

        # Calculating full and local map sizes
        self.map_size = args.map_size_cm // args.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w = int(self.full_w / args.global_downscaling)
        self.local_h = int(self.full_h / args.global_downscaling)

        # Initializing full and local map
        self.full_map = torch.zeros(nc, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(nc, self.local_w,
                                self.local_h).float().to(self.device)

        self.local_ob_map = np.zeros((self.local_w,
                                self.local_h))

        self.local_ex_map = np.zeros((self.local_w,
                                self.local_h))

        self.target_edge_map = np.zeros((self.local_w,
                                self.local_h))

        self.target_point_map = np.zeros((self.local_w,
                                self.local_h))

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        self.tv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
        

        # Initial full and local pose
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.local_pose = torch.zeros(3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros(3)

        # Local Map Boundaries
        self.lmb = np.zeros(4).astype(int)

        self.eve_angle = 0

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros(7)

        self.init_map_and_pose()

        # ------------------------------------------------------------------
        ##### Semantic Mapping init
        # ------------------------------------------------------------------
        self.sem_map_module = Semantic_Mapping(args).to(self.device)
        self.sem_map_module.eval()

        # ------------------------------------------------------------------
        ##### Pred Map init
        # ------------------------------------------------------------------
        self.color_map = None
        self.explo_area_map = None

        # ------------------------------------------------------------------
        ##### EXIT Rotation state init
        # ------------------------------------------------------------------
        self.EXIT = False
        self.Perception_PR = 0
        self.Max_Perception_Angle = 360
        self.count_rerotation = 0
        self.angle_score = 0
        self.is_Frontier = True

        self.Find_Goal = False





    def reset(self) -> None:

        self.init_map_and_pose()

        self.l_step = 0
        self.last_action = None
        self.col_width = 1

        self.episode_n += 1
        self.collision_n = 0
        self.replan_count = 0
        self.replan_flag = 0
        self.stop = 0

        self.eve_angle = 0

        self.stair_flag = 0

        self.goal_name = None
        self.goal_id = None

        self.curr_loc = [self.args.map_size_cm / 100.0 / 2.0,
                         self.args.map_size_cm / 100.0 / 2.0, 0.]

        map_shape = (self.map_size, self.map_size)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        # print(self.episode_n)
        self.Find_Goal = False
        self.Start_Location = None
        self.Path_Length = 1e-5

        self.last_planning_window = None
        self.last_sem_map_vis = None
        self.count_windows = 0
        self.count_masks = 0
        self.count_full_masks = 0
        self.pre_g_points = None
        self.diffusion_output_local = None
        self.diffusion_output_global = None
        self.new_long_term_goal_point = None

        self.new_long_term_goal = None
        self.new_pred_goal_map = None

    def set_legend(self):
        save_legend(self.object_category)
        self.legend = cv2.imread('img/legend.png')
        h, w = self.legend.shape[0], self.legend.shape[1]
        self.legend = cv2.resize(self.legend,
                                    (int(w*980/h), 980),
                                    interpolation=cv2.INTER_NEAREST)
        lx, ly = self.legend.shape[0], self.legend.shape[1]
        if self.vis_image is not None:
            self.vis_image[50:, 1165+500:, :] = 255
            try:
                self.vis_image[50:50 + lx, 1165+500:1165+500 + ly, :] = self.legend
            except:
                logging.info("legend error!")
        

    def mapping(self, observations: Observations):

        # ------------------------------------------------------------------
        ##### At first step, get the object name and init the visualization
        # ------------------------------------------------------------------
        if self.l_step == 0:
            self.last_sim_location = [observations['gps'][0], observations['gps'][1], observations['compass'][0]]
            self.local_pose[2] = observations['compass'][0]* 57.29577951308232

            self.local_pose[2] = torch.fmod(self.local_pose[2] - 180.0, 360.0) + 180.0
            self.local_pose[2] = torch.fmod(self.local_pose[2] + 180.0, 360.0) - 180.0

            actions = torch.randn(1, 2)*6
            cpu_actions = nn.Sigmoid()(actions).cpu().numpy().squeeze()
            global_goals = [int(cpu_actions[0] * self.local_w),
                             int(cpu_actions[1] * self.local_h)]
            self.global_goals = [min(global_goals[0], int(self.local_w - 1)),
                             min(global_goals[1], int(self.local_h - 1))] 
            
            if 'objectgoal' not in observations:
                possible_cats = list(np.arange(6))
                goal_idx = np.random.choice(possible_cats)
                self.goal_id = goal_idx
                for key, value in gibson_coco_categories.items():
                    if value == goal_idx:
                        self.goal_name = key

                if self.args.visualize or self.args.print_images:
                    self.vis_image = vu.init_vis_image(self.goal_name, 0)

            else:
                if 'objectnav_mp3d' in self.args.task_config:
                    self.goal_name = self.object_category[observations['objectgoal'][0]]
                elif 'objectnav_hm3d' in self.args.task_config:
                    self.goal_name = self.object_category[coco_categories_hm3d2mp3d[observations['objectgoal'][0]]]
                self.goal_id = observations['objectgoal'][0]

                if self.args.visualize or self.args.print_images:
                    if 'objectnav_mp3d' in self.args.task_config:
                        self.vis_image = vu.init_vis_image(self.object_category[observations['objectgoal'][0]], 0)
                    elif 'objectnav_hm3d' in self.args.task_config:
                        self.vis_image = vu.init_vis_image(self.object_category[coco_categories_hm3d2mp3d[observations['objectgoal'][0]]], 0)
            # print("objectgoal: ", observations['objectgoal'])

            if 'objectgoal' in observations:
                if observations['objectgoal'][0] == 3:
                    return 0
        # ------------------------------------------------------------------


        # ------------------------------------------------------------------
        ##### Preprocess the observation
        # ------------------------------------------------------------------
        rgb = observations['rgb'].astype(np.uint8)
        depth = observations['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        if self.args.use_sam:
            isChanged = (len(self.object_category) != len(object_category))
            self.object_category = deepcopy(object_category)
            if isChanged:
                self.GSAM.set_text(object_category[:-1])
                self.set_legend()

            obs = self._preprocess_obs(state) 
        else:
            obs = self._preprocess_obs_rednet(state) 

        obs = torch.from_numpy(obs).float().to(self.device)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        ##### local semantic map updating
        # ------------------------------------------------------------------
        poses = torch.from_numpy(np.asarray(self.get_pose_change(observations['gps'], observations['compass']))).float().to(self.device)
        
        points, self.local_map, self.local_map_stair, self.local_pose = \
            self.sem_map_module(obs.unsqueeze(0), poses.unsqueeze(0), self.local_map.unsqueeze(0), self.local_pose.unsqueeze(0), self.eve_angle)


        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs + self.origins
        self.local_map[2, :, :].fill_(0.)  # Resetting current location channel
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]
        self.local_map[2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        ##### Outlines for stucking
        # ------------------------------------------------------------------
        if self.replan_count > self.args.num_local_steps-5 and torch.any(self.local_map[18, :, :]>0):
            self.replan_flag = 1

        # clear the obstacle during the stairs
        # if (torch.any(self.local_map[18, loc_r-10:loc_r+10, loc_c-10:loc_c+10] > 0) and self.replan_flag) or self.local_map[18, loc_r, loc_c] > 0:
        # if  self.replan_flag or self.local_map[18, loc_r-1, loc_c-1] > 0.5:
        #     self.stair_flag = 1
        # # else:
        # #     self.stair_flag = 0

        # if self.stair_flag:
        #     self.local_map[0, :, :] = self.local_map_stair[0, :, :]

        if self.replan_flag:
        #     # must > 0
            self.local_map[0, :, :][self.local_map[18, :, :] > 0] = 0
        # ------------------------------------------------------------------

        
        frontier_score_list = []


        # ------------------------------------------------------------------
        ##### Global Policy
        # ------------------------------------------------------------------
        if self.l_step % self.args.num_local_steps == self.args.num_local_steps - 1:
            # ------------------------------------------------------------------
            ##### Random Policy if no frontiers 
            # ------------------------------------------------------------------
            actions = torch.randn(1, 2)*1
            cpu_actions = nn.Sigmoid()(actions).cpu().numpy().squeeze()
            global_goals = [int(cpu_actions[0] * self.local_w),
                            int(cpu_actions[1] * self.local_h)]
            self.global_goals = [min(global_goals[0], int(self.local_w - 1)),
                            min(global_goals[1], int(self.local_h - 1))] 
                # print("self.global_goals: ", self.global_goals)

            # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        ##### For every global step, update the full maps
        # ------------------------------------------------------------------
        self.full_map[:, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]] = \
            self.local_map
        self.full_pose = self.local_pose + \
            torch.from_numpy(np.asarray(self.origins)).to(self.device).float()

        locs = self.full_pose.cpu().numpy()
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
                                        (self.local_w, self.local_h),
                                        (self.full_w, self.full_h))

        self.planner_pose_inputs[3:] = self.lmb
        self.origins = [self.lmb[2] * self.args.map_resolution / 100.0,
                    self.lmb[0] * self.args.map_resolution / 100.0, 0.]

        self.local_map = self.full_map[:,
                                self.lmb[0]:self.lmb[1],
                                self.lmb[2]:self.lmb[3]]
        self.local_pose = self.full_pose - \
            torch.from_numpy(np.asarray(self.origins)).to(self.device).float()

        if self.replan_count > self.args.num_local_steps-5 or self.collision_n > self.args.num_local_steps - 5:
            self.collision_n = 0
            self.local_map.fill_(0.)
        else:
            self.collision_n = 0
        # ------------------------------------------------------------------
        

    
    def act(self, goal_points: list, )-> Dict[str, int]:
        # ------------------------------------------------------------------
        ##### Update long-term goal if target object is found
        ##### Otherwise, use the VLM to select the goal
        # ------------------------------------------------------------------
        # Rotating
        # if is_rotating:
        #     action = 3
        #     self.last_action = action
        #     self.l_step += 1
        #     return action

        found_goal = 0
    
        local_goal_maps = np.zeros((self.local_w, self.local_h)) 

        local_goal_maps[goal_points[0],goal_points[1]] = 1
            # print("Don't Find the edge")

        if 'objectnav_mp3d' in self.args.task_config:
            cn = self.goal_id + 4
        elif 'objectnav_hm3d' in self.args.task_config:
            cn = coco_categories[self.goal_id] + 4
        if self.args.not_explore == 1:
            if self.local_map[cn, :, :].sum() != 0.:
                logging.info("==========> Find Goal!")
                cat_semantic_map = self.local_map[cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                if cn == 9:
                    cat_semantic_scores = cv2.dilate(cat_semantic_scores, self.tv_kernel)
                local_goal_maps = self.find_big_connect(cat_semantic_scores)
                found_goal = 1
                self.Find_Goal = True
     
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        ##### Take action based on the goal
        # ------------------------------------------------------------------
        planner_inputs = {}
        # self.planner_pose_inputs[3:] = [0, self.local_w, 0, self.local_h]
        planner_inputs['map_pred'] = self.local_map[0, :, :].cpu().numpy()
        planner_inputs['exp_pred'] = self.local_map[1, :, :].cpu().numpy()
        planner_inputs['pose_pred'] = self.planner_pose_inputs
        planner_inputs['goal'] = local_goal_maps  # global_goals[e]
        planner_inputs['map_target'] = self.target_point_map  # global_goals[e]
        planner_inputs['new_goal'] = (self.l_step % self.args.num_local_steps - 1) == 0
        planner_inputs['found_goal'] = self.Find_Goal
        if self.args.visualize or self.args.print_images:
            planner_inputs['map_edge'] = self.target_edge_map
            self.local_map[-1, :, :] = 1e-5
            planner_inputs['sem_map_pred'] = self.local_map[4:, :,
                                            :].argmax(0).cpu().numpy()
        # full_map
        full_planner_inputs = {}
        full_planner_inputs['map_pred'] = self.full_map[0, :, :].cpu().numpy()
        full_planner_inputs['exp_pred'] = self.full_map[1, :, :].cpu().numpy()
        full_planner_inputs['pose_pred'] = self.planner_pose_inputs
        full_planner_inputs['lmb'] = self.lmb
        # full_planner_inputs['goal'] = self.goal_map  
        full_planner_inputs['found_goal'] = self.Find_Goal
        if self.args.visualize or self.args.print_images:
            fvlm = torch.clone(self.full_map[4:, :, :])
            fvlm[-1] = 1e-5
            full_planner_inputs['sem_map_pred'] = fvlm.argmax(0).cpu().numpy() # [480,480]
        

        # if self.args.visualize or self.args.print_images:
        #     self._visualize(planner_inputs, action)
        
        # # 导航
        # if self.args.not_explore == 1:
        #     self.update_pred_local_and_full_map(planner_inputs, full_planner_inputs)
        #     self._visualize(planner_inputs, full_planner_inputs)
        # # 生成
        # else:
        #     self._visualize(planner_inputs, full_planner_inputs)
        #     self._draw_semmap(planner_inputs, full_planner_inputs)
        if self.args.not_explore == 0:
            self._visualize(planner_inputs, full_planner_inputs)
            self._draw_semmap(planner_inputs, full_planner_inputs)


        
        action = self._plan(planner_inputs)
        # print("self.l_step: ", self.l_step)
        # print("action: ", action)

        self.last_action = action
        self.l_step += 1

        return action

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        # if planner_inputs["new_goal"]:
        #     self.collision_map = np.zeros(self.visited.shape)

        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        exp_pred = np.rint(planner_inputs['exp_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        # print("pose_pred:",planner_inputs['pose_pred'])
        # print("start_x:",start_x)
        # print("start_y:",start_y)
        # print("start_o:",start_o)

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        # if args.visualize or args.print_images:
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                            self.visited_vis[gx1:gx2, gy1:gy2])
        if self.l_step == 0:
            self.Start_Location = last_start
        self.Path_Length += pu.get_l2_distance(last_start[0],start[0],last_start[1],start[1])


        # Collision check
        if self.last_action == 1 and not planner_inputs["new_goal"]:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                self.collision_n += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)


        # Deterministic Local Policy
        if stop and planner_inputs['found_goal'] == 1:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            
            # print("start_o:",start_o)
            
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            ## add the evelution angle
            eve_start_x = int(5 * math.sin(angle_st_goal) + start[0])
            eve_start_y = int(5 * math.cos(angle_st_goal) + start[1])
            if eve_start_x >= map_pred.shape[0]: eve_start_x = map_pred.shape[0]-1
            if eve_start_y >= map_pred.shape[0]: eve_start_y = map_pred.shape[0]-1 
            if eve_start_x < 0: eve_start_x = 0 
            if eve_start_y < 0: eve_start_y = 0 
            if exp_pred[eve_start_x, eve_start_y] == 0 and self.eve_angle > -60:
                action = 5
                self.eve_angle -= 30
            elif exp_pred[eve_start_x, eve_start_y] == 1 and self.eve_angle < 0:
                action = 4
                self.eve_angle += 30
            elif relative_angle > self.args.turn_angle:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle:
                action = 2  # Left
            elif relative_angle > self.args.turn_angle / 2.:
                action = 7  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 6  # Left
            else:
                action = 1  # Forward
        
        # print("action:",action)

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        # print("grid: ", grid.shape)

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        # traversible = grid[x1:x2, y1:y2] != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self.kernel) == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        if replan:
            self.replan_count += 1
            print("false: ", self.replan_count)
        else:
            self.replan_count = 0

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        # print("obs: ", obs.shape)
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        semantic = obs[:,:,4:5].squeeze()

        # BGR to RGB
        self.rgb_vis = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                                  (640, 480), interpolation=cv2.INTER_NEAREST)
        # print("obs: ", semantic.shape)
        if args.use_gtsem:
            self.semantics_vis = self.rgb_vis
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
            for i in range(16):
                sem_seg_pred[:,:,i][semantic == i+1] = 1
        else: 
            semantic_output = self._get_sem_pred(
                rgb.astype(np.uint8), depth, use_seg=use_seg)
            sam_semantic_pred = semantic_output['sam_semantic_pred']
            sam_all_cls = convert_SAM(sam_semantic_pred, self.object_category)
            sem_seg_pred = sam_all_cls

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_obs_rednet(self, obs, use_seg=True):
        args = self.args
        # print("obs: ", obs.shape)
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        red_semantic_pred, semantic_pred = self._get_sem_pred_rednet(
            rgb.astype(np.uint8), depth, use_seg=use_seg)

        if 'objectnav_hm3d' in args.task_config:
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))   
            for i in range(0, 15):
                # print(mp_categories_mapping[i])
                sem_seg_pred[:,:,i][red_semantic_pred == mp_categories_mapping[i]] = 1
        elif 'objectnav_mp3d' in args.task_config:
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 20 + 1))   
            for i in range(0, 20):
                # print(mp_categories_mapping[i])
                sem_seg_pred[:,:,i][red_semantic_pred == mp_categories_mapping21[i]] = 1

        sem_seg_pred[:,:,0][semantic_pred[:,:,0] == 0] = 0
        sem_seg_pred[:,:,1][semantic_pred[:,:,1] == 0] = 0
        sem_seg_pred[:,:,2][semantic_pred[:,:,2] == 1] = 1
        sem_seg_pred[:,:,3][semantic_pred[:,:,3] == 0] = 0
        sem_seg_pred[:,:,4][semantic_pred[:,:,4] == 1] = 1
        sem_seg_pred[:,:,5][semantic_pred[:,:,5] == 1] = 1
        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), depth, use_seg=use_seg)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        # print("depth origin: ", depth.shape)
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + (max_d-min_d) * depth * 100.0
        # depth = depth*1000.

        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
        if use_seg:
            # # save rgb and depth
            # skimage.io.imsave("current_rgb.png", rgb)
            # skimage.io.imsave("current_depth.png", (np.repeat(depth, 3, axis=2) * 255).astype(np.uint8))
            
            self.semantics_vis = None
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            with torch.no_grad():
                # print(image.shape, depth.shape) # torch.Size([1, 480, 640, 3]) torch.Size([1, 480, 640, 1])
                try:
                    rgb_Image = Image.fromarray(rgb).convert('RGB')
                    
                    sam_semantic_pred = self.GSAM.predict(rgb_Image) # (N, 1, 480, 640), we need (480, 640, 16)
                    self.semantics_vis = self.GSAM.get_vis(rgb_Image, sam_semantic_pred)
                    # plt.figure(figsize=(10, 10))
                    # for mask in sam_semantic_pred[0]:
                    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                    # for box, label in zip(sam_semantic_pred[1], sam_semantic_pred[2]):
                    #     show_box(box.numpy(), plt.gca(), label)
                    # output_dir = "./tmp_panorama/"
                    # plt.axis('off')
                    # plt.savefig(
                    #     os.path.join(output_dir, f"grounded_sam_output_{self.t}.jpg"), 
                    #     bbox_inches="tight", dpi=300, pad_inches=0.0
                    # )
                    # self.t += 1
                    # save_mask_data("Grounded_SAM/output_sam/", sam_semantic_pred[0], sam_semantic_pred[1], sam_semantic_pred[2])
                except Exception as ex:
                    print(f"[SAM]: no object detected: {ex}")
                    sam_semantic_pred = None
                    
                if self.semantics_vis is None:
                    self.semantics_vis = self.rgb_vis
        else:
            raise NotImplementedError
        outputs = {
            "sam_semantic_pred": sam_semantic_pred,
        }
        return outputs
    
    def _get_sem_pred_rednet(self, rgb, depth, use_seg=True):
        if use_seg:
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            red_semantic_pred = self.red_sem_pred(image, depth).squeeze().cpu().detach().numpy()

            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return red_semantic_pred, semantic_pred

    def get_pose_change(self, gps, compass):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        # print("gps: ", gps)
        # print("compass: ", compass)
        curr_sim_pose = [gps[0],-gps[1], compass[0]]
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def find_big_connect(self, image):
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

    def remove_small_points(self, local_ob_map, image, threshold_point, pose):
        # print("goal_cat_id: ", goal_cat_id)
        # print("sem: ", sem.shape)
        selem = skimage.morphology.disk(1)
        traversible = skimage.morphology.binary_dilation(
            local_ob_map, selem) != True
        # traversible = 1 - traversible
        planner = FMMPlanner(traversible)
        goal_pose_map = np.zeros((local_ob_map.shape))
        pose_x = int(pose[0].cpu()) if int(pose[0].cpu()) < self.local_w-1 else self.local_w-1
        pose_y = int(pose[1].cpu()) if int(pose[1].cpu()) < self.local_w-1 else self.local_w-1
        goal_pose_map[pose_x, pose_y] = 1
        # goal_map = skimage.morphology.binary_dilation(
        #     goal_pose_map, selem) != True
        # goal_map = 1 - goal_map
        planner.set_multi_goal(goal_pose_map)

        img_label, num = measure.label(image, connectivity=2, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        # print("img_label.dtype: ", img_label.dtype) # 480*480
        Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
        Goal_point = np.zeros(img_label.shape)
        dict_cost = {}
        for i in range(1, len(props)):
            # print("area: ", props[i].area)
            # dist = pu.get_l2_distance(props[i].centroid[0], pose[0], props[i].centroid[1], pose[1])
            dist = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])] * 5
            # dist_s = 8 if dist < 300 else 0
            
            cost = dist

            if props[i].area > threshold_point and dist > 50 and dist < 500:
                dict_cost[i] = cost
        
        if dict_cost:
            dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=False)
            
            # print(dict_cost)
            for i, (key, value) in enumerate(dict_cost):
                # print(i, key)
                Goal_edge[img_label == key + 1] = 1
                Goal_point[int(props[key].centroid[0]), int(props[key].centroid[1])] = i+1 #
                if i == 3:
                    break

        return Goal_edge, Goal_point

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.full_map[2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0


        self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
                                            (self.local_w, self.local_h),
                                            (self.full_w, self.full_h))

        self.planner_pose_inputs[3:] = self.lmb
        self.origins = [self.lmb[2] * self.args.map_resolution / 100.0,
                    self.lmb[0] * self.args.map_resolution / 100.0, 0.]

        self.local_map = self.full_map[:,
                                self.lmb[0]:self.lmb[1],
                                self.lmb[2]:self.lmb[3]]
        self.local_pose = self.full_pose - \
            torch.from_numpy(np.array(self.origins)).to(self.device).float()

        self.local_ob_map = np.zeros((self.local_w,
                                self.local_h))

        self.local_ex_map = np.zeros((self.local_w,
                                self.local_h))

        self.target_edge_map = np.zeros((self.local_w,
                                self.local_h))

        self.target_point_map = np.zeros((self.local_w,
                                self.local_h))

    def get_frontier_boundaries(self, frontier_loc, frontier_sizes, map_sizes):
        loc_r, loc_c = frontier_loc
        local_w, local_h = frontier_sizes
        full_w, full_h = map_sizes

        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
 
        return [int(gx1), int(gx2), int(gy1), int(gy2)]
    



    def find_closest_color(self, pixel, color_pal):
        distances = np.sqrt(np.sum((color_pal - pixel) ** 2, axis=1))
        return np.argmin(distances)

    def find_intersection(self, start, end, gx1, gx2, gy1, gy2):
        x1, y1 = start
        x2, y2 = end

        dx = x2 - x1
        dy = y2 - y1

        def compute_intersection(x, y, dx, dy, boundary):
            t = (boundary - x) / dx if dx != 0 else np.inf
            return (x + t * dx, y + t * dy) if 0 <= t <= 1 else None

        intersections = []
        for boundary in [gx1, gx2]:
            intersection = compute_intersection(x1, y1, dx, dy, boundary)
            if intersection and gy1 <= intersection[1] <= gy2:
                intersections.append(intersection)
        for boundary in [gy1, gy2]:
            intersection = compute_intersection(y1, x1, dy, dx, boundary)
            if intersection and gx1 <= intersection[1] <= gx2:
                intersections.append((intersection[1], intersection[0]))

        if intersections:
            distances = [np.linalg.norm(np.array(start) - np.array(p)) for p in intersections]
            return intersections[np.argmin(distances)]
        return None

    def find_navigation_target(self, centers, num_points, cluster_density, gx1, gx2, gy1, gy2, start):

        mask = (centers[:, 0] >= gx1) & (centers[:, 0] <= gx2) & \
            (centers[:, 1] >= gy1) & (centers[:, 1] <= gy2)
        filtered_centers = centers[mask]
        filtered_num_points = num_points[mask]
        filtered_density = cluster_density[mask]
        
        if len(filtered_centers) > 0:
            start = np.array(start)
            density_scores = 1 / (filtered_density + 1e-9) 

            num_points_scores = filtered_num_points
            
            distances = np.linalg.norm(filtered_centers - start, axis=1)
            distance_scores = 1 / (distances + 1e-9) 
            
            weight_density = 0.5  
            weight_num_points = 0.4  
            weight_distance = 0.1 
            
            total_scores = (
                weight_density * density_scores +
                weight_num_points * num_points_scores +
                weight_distance * distance_scores
            )
            
            best_index = np.argmax(total_scores)
            return tuple(filtered_centers[best_index])
        else:
            start = np.array(start)
            distances = np.linalg.norm(centers - start, axis=1)
            nearest_index = np.argmin(distances)
            nearest_center = tuple(centers[nearest_index])
            
            intersection = self.find_intersection(start, nearest_center, gx1, gx2, gy1, gy2)
            if intersection:
                return intersection
            else:
                print("No intersection is found, return to the nearest cluster center point.")
                return nearest_center
    
    def remove_sparse_coordinates(self, coords, radius=2, min_neighbors=6):
        if len(coords) == 0:
            return coords

        points = np.array(coords)

        tree = KDTree(points)

        neighbor_indices = tree.query_ball_point(points, r=radius)
        
        neighbor_counts = np.array([len(indices) - 1 for indices in neighbor_indices])

        mask = neighbor_counts >= min_neighbors
        filtered_points = points[mask]
        
        return filtered_points

    def update_pred_local_and_full_map(self, inputs, f_inputs):
        """Generate semmap and save."""
        args = self.args

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']

        f_map_pred = f_inputs['map_pred']
        f_exp_pred = f_inputs['exp_pred']

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        # goal = inputs['goal']

        pos = (
            int((start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0]),
            int((map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1])
        )

        planning_window = [gx1, gx2, gy1, gy2]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        new_pred_local_goal_map = None

        # print("planning_window:",planning_window)#[480*480]

        # Record Previous Step Window
        # if planning_window == self.last_planning_window or self.timestep == 1: # timestep==1是为了记录第一步的last_sem_map_vis
        sem_map = inputs['sem_map_pred'] #[480,480]
        
        # self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        sem_map += 5
        sem_map[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 14
        # if int(self.stg[0]) < self.local_w and int(self.stg[1]) < self.local_h:
        #     sem_map[int(self.stg[0]),int(self.stg[1])] = 15

        no_cat_mask = sem_map == args.num_sem_categories + 4
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        # Del Traj
        sem_map[sem_map == 3] = 2
        # Del long-time goal
        sem_map[sem_map == 4] = 0

        # Draw goal dot
        # selem = skimage.morphology.disk(4)
        # goal_mat = 1 - skimage.morphology.binary_dilation(
        #     goal, selem) != True
        
        # goal_mask = goal_mat == 1
        # sem_map[goal_mask] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST) ## build local map
        self.last_sem_map_vis = sem_map_vis

        # Record mask every local_steps 20 (begin from 1st step) 保留这段代码
        # if self.timestep % args.num_local_steps == 1:
        #     self.count_masks += 1
        #     mask_img = np.zeros_like(sem_map_vis)                
        #     lower_white = np.array([252, 252, 252])  
        #     upper_white = np.array([255, 255, 255])
        #     mask = cv2.inRange(sem_map_vis, lower_white, upper_white)
        #     white_only = cv2.bitwise_and(sem_map_vis, sem_map_vis, mask=mask)
        #     mask_img[mask != 0] = white_only[mask != 0]
            
        #     fn = '{}/semmap_mask/eps_{}/{}.jpg'.format(
        #         dump_dir, self.episode_n - 1,
        #         self.count_masks)
        #     cv2.imwrite(fn, mask_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # else: #(self.count_windows == 0 and planning_window != self.last_planning_window) or (planning_window != self.last_planning_window and self.count_windows > 0):
        # self.count_windows += 1
        # sem_map_vis = self.last_sem_map_vis
        # inputs['sem_map_pred'] += 5

        # cv2.imwrite('sem_map_output_origin.png',sem_map_vis)

        # # Bulid full map
        f_sem_map = f_inputs['sem_map_pred'] #[480,480]
        f_sem_map += 5
        f_sem_map[self.collision_map == 1] = 14

        f_no_cat_mask = f_sem_map == args.num_sem_categories + 4
        f_map_mask = np.rint(f_map_pred) == 1
        f_exp_mask = np.rint(f_exp_pred) == 1
        # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1


        f_sem_map[f_no_cat_mask] = 0
        f_m1 = np.logical_and(f_no_cat_mask, f_exp_mask)
        f_sem_map[f_m1] = 2

        f_m2 = np.logical_and(f_no_cat_mask, f_map_mask)
        f_sem_map[f_m2] = 1

        color_pal = [int(x * 255.) for x in color_palette]
        f_sem_map_vis = Image.new("P", (f_sem_map.shape[1],
                                    f_sem_map.shape[0]))
        f_sem_map_vis.putpalette(color_pal)
        f_sem_map_vis.putdata(f_sem_map.flatten().astype(np.uint8))
        f_sem_map_vis = f_sem_map_vis.convert("RGB")
        f_sem_map_vis = np.flipud(f_sem_map_vis)
        f_sem_map_vis = f_sem_map_vis[:, :, [2, 1, 0]]

        f_sem_map_vis_origin = f_sem_map_vis
        
        # f_sem_map_vis = self.crop_and_resize(f_sem_map_vis)
        f_sem_map_vis = self.only_crop(f_sem_map_vis) ## build full map
        sem_map_vis = cv2.resize(sem_map_vis, (256, 256),
                                interpolation=cv2.INTER_NEAREST) ## build local map
        
        transform = transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

#################################################===Diffusion Process===#################################################
        if self.l_step > 90 and self.l_step % 50 == 0 and inputs['found_goal'] == 0:
            print("TimeStep {}, local map diffusion process...".format(self.l_step))
            sem_map_vis_tmp = deepcopy(sem_map_vis)
            crop_size,sem_map_vis = smart_crop_resize(cv2.cvtColor(sem_map_vis, cv2.COLOR_RGB2BGR))
            x1, x2, y1, y2 = crop_size
            target_size = x2 - x1

            mask = np.full(sem_map_vis.shape[:2], 255, dtype=np.uint8)
            white_threshold = np.array([245, 245, 245])
            white_mask = np.all(sem_map_vis > white_threshold, axis=-1)
            mask[white_mask] = 0 
            
            mask = transform(Image.fromarray(mask))
            mask = mask.unsqueeze(0)

            sem_map_vis_n = sem_map_vis.astype(np.float32) / 255.
            gray_image = rgb2gray(np.array(tensor_to_image()(sem_map_vis_n)))
            edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=2.)))
            gray_image = image_to_tensor()(Image.fromarray(gray_image))
            Y_GT, X_GT, X_LQ = sem_map_vis_n,gray_image,edge ##completed grayscale and edge images

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


            noisy_state = self.sde.noise_state(Y_GT * mask) # *mask的
            noisy_states = self.S_sde.noise_state(X_LQ * mask) # * mask
            self.model.feed_data(noisy_state, Y_GT * mask, Y_GT, mask, self.S_sde, X_GT,  X_LQ * mask)
            self.model.test(self.sde, save_states=True, GT = Y_GT, mask = mask, \
                            S_sde = self.S_sde, S_GT = X_GT, S_LQ = noisy_states, dis = self.model.dis, save_dir=None)
            
            # toc = time.time()#
            # test_times.append(toc - tic)
            visuals = self.model.get_current_visuals()
            SR_img = visuals["Output"]
            output = util.tensor2img(SR_img.squeeze())  # uint8
            LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
            GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8

            resized_output = cv2.resize(output, (target_size, target_size), 
                              interpolation=cv2.INTER_LANCZOS4)
            sem_map_vis_tmp[x1:x2,y1:y2] = resized_output
            
            # cv2.imwrite('sem_map_output_origin.png', output)
            output = cv2.resize(sem_map_vis_tmp, (480, 480), interpolation=cv2.INTER_NEAREST)
            # output = cv2.copyMakeBorder(output, 240, 240, 240, 240, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # black_pixels = np.all(output <= 35, axis=-1)
            # output[black_pixels] = [255, 255, 255]

            
            # color_pal = [int(x * 255.) for x in color_palette]
            color_pal_resize = np.array(color_pal).reshape(-1, 3)
            # global_map_prediction_mapped = np.zeros((960, 960), dtype=np.uint8)
            # for y in range(960):
            #     for x in range(960):
            #         pixel = output[y, x]
            #         closest_color_index = self.find_closest_color(pixel, color_pal_resize)
            #         global_map_prediction_mapped[y, x] = closest_color_index
            # global_map_prediction_mapped = np.flipud(global_map_prediction_mapped)


            # # print(f_sem_map)
            # cv2.imwrite('f_sem_map.png', f_map_test)
            # exit()

            lmb = f_inputs['lmb']

            # local_sem_map = f_sem_map_vis_origin[lmb[0]:lmb[1], lmb[2]:lmb[3], :]
            local_sem_map = sem_map_vis.copy()
            
            # output_z = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # global_goals = np.argwhere(output == color_pal_resize[self.goal_cat+5]) # goal_cat是基于map_category_names的映射  
            if 'objectnav_hm3d' in self.args.task_config:
                matches = np.where((output == color_pal_resize[coco_categories[self.goal_id] + 5]).all(axis=2))
            elif 'objectnav_mp3d' in self.args.task_config:
                matches = np.where((output == color_pal_resize[self.goal_id + 5]).all(axis=2))
            global_goals = []
            for y, x in zip(matches[0], matches[1]):
                global_goals.append([y, x])
            global_goals = np.array(global_goals)


            # cv2.circle(output, pos, radius=8, color=(0, 255, 0), thickness=-1)  # 绿色点(start_x,start_y)
            if len(global_goals) > 0:
                global_goal,num_points,cluster_density = self.cluster.predict(global_goals)
                
                for global_goal2 in global_goals:
                    center2 = (global_goal2[1],global_goal2[0])
                    # cv2.circle(local_sem_map, center2, radius=1, color=(255, 0, 0), thickness=-1)  # 蓝色点
                    cv2.circle(output, (global_goal2[1],global_goal2[0]), radius=1, color=(255, 0, 0), thickness=-1)  # 蓝色点

                if len(global_goal) > 0:
                    for center in global_goal:
                        local_center = (center[1], center[0])
                        # cv2.circle(local_sem_map, local_center, radius=5, color=(0, 0, 255), thickness=-1)  # 红色点
                        cv2.circle(output, (center[1],center[0]), radius=5, color=(0, 0, 255), thickness=-1)  # 红色点
                else:
                    distances = np.linalg.norm(global_goals - pos, axis=1)
                    closest_index = np.argmin(distances)
                    t_point = global_goals[closest_index]
                    cv2.circle(output, (t_point[1],t_point[0]), radius=5, color=(0, 0, 255), thickness=-1)

                if len(global_goal)>0:
                    fullmap_new_goal_point = self.find_navigation_target(np.array(global_goal),np.array(num_points),np.array(cluster_density), \
                                                                          lmb[0],lmb[1],lmb[2],lmb[3], pos)
                    self.new_long_term_goal_point = (fullmap_new_goal_point[0],fullmap_new_goal_point[1])
                    
                    # for center in centers:
                    #     cv2.circle(image, tuple(center), radius=5, color=(0, 0, 255), thickness=-1)
                    new_pred_local_goal_map = np.zeros((inputs['goal'].shape[0],inputs['goal'].shape[1]))
                    new_pred_local_goal_map[int(self.new_long_term_goal_point[0]),int(self.new_long_term_goal_point[1])] = 1

                    self.new_pred_goal_map = np.flipud(new_pred_local_goal_map)
                
                elif len(global_goals) > 0:
                    fullmap_new_goal_point = tuple(t_point)
                    self.new_long_term_goal_point = (fullmap_new_goal_point[0],fullmap_new_goal_point[1])
                    
                    # for center in centers:
                    #     cv2.circle(image, tuple(center), radius=5, color=(0, 0, 255), thickness=-1)
                    new_pred_local_goal_map = np.zeros((inputs['goal'].shape[0],inputs['goal'].shape[1]))
                    new_pred_local_goal_map[int(self.new_long_term_goal_point[0]),int(self.new_long_term_goal_point[1])] = 1

                    self.new_pred_goal_map = np.flipud(new_pred_local_goal_map)
                    
            # self.diffusion_output = local_sem_map
            self.diffusion_output_local = output
            # self.diffusion_output = cv2.resize(self.diffusion_output, (480, 480),
            #                  interpolation=cv2.INTER_NEAREST)
            
            
                # print("new long-term goal:",self.new_long_term_goal_point)

        # and inputs['found_goal'] == 0
        if self.pre_g_points is not None and self.l_step % 20 == 0:
            if calculate_distance(self.pre_g_points, pos) <= 5:
                actions = np.random.rand(1, 2).squeeze()*(480 - 1)
                self.new_long_term_goal_point = (int(actions[0]), int(actions[1]))
                new_pred_local_goal_map = np.zeros((inputs['goal'].shape[0],inputs['goal'].shape[1]))
                new_pred_local_goal_map[self.new_long_term_goal_point[0],self.new_long_term_goal_point[1]] = 1

                self.new_pred_goal_map = np.flipud(new_pred_local_goal_map)

        self.pre_g_points = pos


        if self.new_pred_goal_map is not None and inputs['found_goal'] == 0:
            inputs['goal'] = self.new_pred_goal_map

        self.last_planning_window = planning_window
    

    def crop_and_resize(self, image, output_size=(256, 256)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV) 
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Warning: No contours found, the image might be completely white.")
            resized_image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            return resized_image
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        original_aspect_ratio = w / h
        target_aspect_ratio = output_size[0] / output_size[1]
        
        if original_aspect_ratio > target_aspect_ratio:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            padded_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        else:
            new_width = h
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            padded_image = cv2.copyMakeBorder(cropped_image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        resized_image = cv2.resize(padded_image, output_size, interpolation=cv2.INTER_NEAREST)
        # print(padded_image.shape)
        return resized_image

    def only_crop(self, image, output_size=(256, 256)):
        
        resized_image = cv2.resize(image, output_size, interpolation=cv2.INTER_NEAREST)
        # print(padded_image.shape)
        return resized_image


    def _draw_semmap(self, inputs, f_inputs):
        """Generate semmap and save."""
        args = self.args

        dump_dir = "{}/mapdata/".format(args.dump_location)
        sem_dir = '{}/semmap/eps_{}/'.format(
            dump_dir, self.episode_n - 1)
        f_sem_dir = '{}/full_semmap/eps_{}/'.format(
            dump_dir, self.episode_n - 1)
        mask_dir = '{}/semmap_mask/eps_{}/'.format(
            dump_dir, self.episode_n - 1)
        full_mask_dir = '{}/full_semmap_mask/eps_{}/'.format(
            dump_dir, self.episode_n - 1)
        if not os.path.exists(sem_dir):
            os.makedirs(sem_dir)
        if not os.path.exists(f_sem_dir):
            os.makedirs(f_sem_dir)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        if not os.path.exists(full_mask_dir):
            os.makedirs(full_mask_dir)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']

        f_map_pred = f_inputs['map_pred']
        f_exp_pred = f_inputs['exp_pred']

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        # goal = inputs['goal']

        planning_window = [gx1, gx2, gy1, gy2]


        # Record Previous Step Window
        # if planning_window == self.last_planning_window or self.timestep == 1:
        sem_map = inputs['sem_map_pred'] #[240,240]
        
        # self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        # sem_map += 5
        sem_map[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 14
        # if int(self.stg[0]) < self.local_w and int(self.stg[1]) < self.local_h:
        #     sem_map[int(self.stg[0]),int(self.stg[1])] = 15

        no_cat_mask = sem_map == args.num_sem_categories + 4
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        # Del Traj
        sem_map[sem_map == 3] = 2
        # Del long-time goal
        sem_map[sem_map == 4] = 0

        # Draw goal dot
        # selem = skimage.morphology.disk(4)
        # goal_mat = 1 - skimage.morphology.binary_dilation(
        #     goal, selem) != True
        
        # goal_mask = goal_mat == 1
        # sem_map[goal_mask] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (256, 256),
                                interpolation=cv2.INTER_NEAREST)
        self.last_sem_map_vis = sem_map_vis
        
        # Bulid full map
        f_sem_map = f_inputs['sem_map_pred'] #[960,960]
        f_sem_map += 5
        f_sem_map[self.collision_map == 1] = 14

        f_no_cat_mask = f_sem_map == args.num_sem_categories + 4
        f_map_mask = np.rint(f_map_pred) == 1
        f_exp_mask = np.rint(f_exp_pred) == 1
        # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1


        f_sem_map[f_no_cat_mask] = 0
        f_m1 = np.logical_and(f_no_cat_mask, f_exp_mask)
        f_sem_map[f_m1] = 2

        f_m2 = np.logical_and(f_no_cat_mask, f_map_mask)
        f_sem_map[f_m2] = 1

        # # Del Traj
        # f_sem_map[f_sem_map == 3] = 2
        # # Del long-time goal
        # f_sem_map[f_sem_map == 4] = 0

        # f_sem_map -= 9

        # color_pal = [int(x * 255.) for x in color_palette]
        # print(sem_map)
        # print(f_sem_map)
        
        f_sem_map_vis = Image.new("P", (f_sem_map.shape[1],
                                    f_sem_map.shape[0]))
        f_sem_map_vis.putpalette(color_pal)
        f_sem_map_vis.putdata(f_sem_map.flatten().astype(np.uint8))
        f_sem_map_vis = f_sem_map_vis.convert("RGB")
        f_sem_map_vis = np.flipud(f_sem_map_vis)
        f_sem_map_vis = f_sem_map_vis[:, :, [2, 1, 0]]
        # f_sem_map_vis = self.crop_and_resize(f_sem_map_vis)
        f_sem_map_vis = self.only_crop(f_sem_map_vis)
        # f_sem_map_vis = cv2.resize(f_sem_map_vis, (256, 256),
        #                         interpolation=cv2.INTER_NEAREST)
        if self.l_step > 498:
            fn = '{}/full_semmap/eps_{}/full_map_{}.jpg'.format(
                    dump_dir, self.episode_n - 1,
                    self.episode_n - 1)
            cv2.imwrite(fn, f_sem_map_vis, [cv2.IMWRITE_JPEG_QUALITY, 100])

        if  self.l_step % args.num_local_steps == 1: # 如何设定保存全局语义地图阈值？
            self.count_masks += 1
            mask_img = np.zeros_like(sem_map_vis)                
            lower_white = np.array([252, 252, 252])  
            upper_white = np.array([255, 255, 255])
            mask = cv2.inRange(sem_map_vis, lower_white, upper_white)
            white_only = cv2.bitwise_and(sem_map_vis, sem_map_vis, mask=mask)
            mask_img[mask != 0] = white_only[mask != 0]
            
            fn = '{}/semmap_mask/eps_{}/{}.jpg'.format(
                dump_dir, self.episode_n - 1,
                self.count_masks)
            cv2.imwrite(fn, mask_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

            # exit()


            # Record full mask every local_steps 20 (begin from 1st step)
            self.count_full_masks += 1
            mask_img = np.zeros_like(f_sem_map_vis)                
            lower_white = np.array([252, 252, 252])  
            upper_white = np.array([255, 255, 255])
            mask = cv2.inRange(f_sem_map_vis, lower_white, upper_white)
            white_only = cv2.bitwise_and(f_sem_map_vis, f_sem_map_vis, mask=mask)
            mask_img[mask != 0] = white_only[mask != 0]
            
            fn = '{}/full_semmap_mask/eps_{}/{}.jpg'.format(
                dump_dir, self.episode_n - 1,
                self.count_full_masks)
            cv2.imwrite(fn, mask_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

            sem_map_vis = self.last_sem_map_vis
            fn = '{}/semmap/eps_{}/{}.jpg'.format(
                    dump_dir, self.episode_n - 1,
                    self.count_masks)
            cv2.imwrite(fn, sem_map_vis, [cv2.IMWRITE_JPEG_QUALITY, 100]) #2025.1.4 在使用全局语义图过程中，我们暂时不画出来

        
        self.last_planning_window = planning_window



    def _visualize(self, inputs, f_inputs):
        """Generate visualization and save."""

        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/eps_{}/'.format(
            dump_dir, self.episode_n - 1)
        if args.not_explore == 1 and args.visualize == 2:
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']

        f_map_pred = f_inputs['map_pred']
        f_exp_pred = f_inputs['exp_pred']

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        goal = inputs['goal']

        sem_map = inputs['sem_map_pred'] #[240,240]
        
        self.vis_image = vu.init_vis_image_diffusion(self.goal_name, self.legend)

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        
        if args.not_explore == 1:
            sem_map += 0
        else:
            sem_map += 5
        
        sem_map[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 14
        # if int(self.stg[0]) < self.local_w and int(self.stg[1]) < self.local_h:
        #     sem_map[int(self.stg[0]),int(self.stg[1])] = 15

        no_cat_mask = sem_map == args.num_sem_categories + 4
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        # Draw goal dot
        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True
        
        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

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
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis
        
        if self.diffusion_output_local is not None:
            self.vis_image[50:530, 1165:1645] = self.diffusion_output_local
            self.vis_image[50:530, 1164+480] = [100,100,100]
            if self.diffusion_output_global is not None:
                self.vis_image[50:530, 1660:1660+480] = self.diffusion_output_global
                self.vis_image[50:530, 1659+480] = [100,100,100]
            
                            
        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        # Draw agent as an arrow
        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1) 
        
        # if args.visualize == 1:
        #     # Displaying the image
        #     cv2.imshow("Thread {}".format(self.rank), self.vis_image)
        #     cv2.waitKey(1)

        if args.visualize == 2:
            # Saving the image
            if args.not_explore==0:
                pass
            fn = '{}/episodes/eps_{}/{}-Vis-{}.jpg'.format(
                dump_dir, self.episode_n - 1,
                self.episode_n - 1, self.l_step)

            cv2.imwrite(fn, self.vis_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    def get_spl(self, success, cur_loc):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        starting_distance = pu.get_l2_distance(self.Start_Location[0],cur_loc[0],self.Start_Location[1],cur_loc[1])
        spl = min(success * starting_distance / self.Path_Length, 1)
        if self.Path_Length <= 1e-3:
            spl = success * 1
        return spl