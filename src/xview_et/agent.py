import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import transforms

from utils.misc import length2mask

# from r2r.agent_cmt import Seq2SeqCMTAgent
from models.vln_model import CustomBERTModel
from models.dark_net import Darknet
from models.qwen_vl_adapter import QwenVLAdapter
from models.ET_haa import ET
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
from transformers import AutoModel, BertTokenizerFast
# import clip
import cv2
import shapely
import shapely.geometry
from shapely.geometry import Polygon, MultiPoint
from utils.logger import write_to_record_file, print_progress, timeSince

def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    tensors = tensors.items()
    for line in tensors:
        print('{}\t{}'.format(*line))

# https://programmerah.com/using-shapely-geometry-polygon-to-calculate-the-iou-of-any-two-quadrilaterals-28395/
def compute_iou(a, b):
    a = np.array(a)  # quadrilateral two-dimensional coordinate representation
    poly1 = Polygon(
        a).convex_hull  # python quadrilateral object, will automatically calculate four points, the last four points in the order of: top left bottom right bottom right top left top
    # print(Polygon(a).convex_hull)  # you can print to see if this is the case

    b = np.array(b)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # Merge two box coordinates to become 8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # contains the smallest polygon point of the two quadrilaterals
    if not poly1.intersects(poly2):  # If the two quadrilaterals do not intersect
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # intersection area
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area)/(union_area-inter_area)  #wrong
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # The source code gives two ways to calculate IOU, the first one is: intersection part / area of the smallest polygon containing two quadrilaterals
            # The second one: intersection/merge (common way to calculate IOU of rectangular box)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def is_default_gpu(opts) -> bool:
    return opts.local_rank == -1 or dist.get_rank() == 0

def get_direction(start,end):
    vec=np.array(end) - np.array(start)
    _angle = 0
    #          90
    #      135    45
    #     180  .    0
    #      225   -45 
    #          270
    if vec[1] > 0: # lng is postive
        _angle = np.arctan(vec[0]/vec[1]) / 1.57*90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0]/vec[1]) / 1.57*90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle+90)%360
    return _angle



class NavCMTAgent:
    def __init__(self, args,  rank=0):
        self.results = {}
        self.losses = [] # For learning agents
        self.args = args
        self.env = []
        self.env_name = ''
        random.seed(1)
        
        # RGB normalization values
        self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
        self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Models
        # self.vln_bert = VLNBertCMT(self.args).cuda()
        # self.critic = Critic(self.args).cuda()
        self.use_qwen_vl = getattr(self.args, "use_qwen_vl", False)

        if self.use_qwen_vl:
            self.qwen_adapter = QwenVLAdapter(
                model_name=self.args.qwen_vl_model,
                proj_dim=self.args.proj_dim,
                lang_cls_dim=49,
                et_frame_dim=49,
                et_frame_tokens=self.args.qwen_et_frame_tokens,
                lstm_frame_channels=512,
                lstm_frame_tokens=49,
                use_layernorm=True,
                l2_norm=self.args.qwen_l2_norm,
                dtype=self.args.qwen_dtype,
                device="cuda",
                verbose=self.args.qwen_verbose,
            ).cuda()
            self.qwen_adapter.set_qwen_trainable(freeze=True)

            self.lang_model = None
            self.vision_model = None
            self.tokenizer = None

            if self.args.use_feature_distill:
                self.tokenizer = BertTokenizerFast.from_pretrained(
                    "/mnt/nfs/dell_15td/avdn/Aerial-Vision-and-Dialog-Navigation-origin/Aerial-Vision-and-Dialog-Navigation-origin/datasets/bert-base-uncased",
                    local_files_only=True,
                )
                self.lang_model = CustomBERTModel(
                    model_name_or_path=self.args.bert_model_path,
                    local_files_only=self.args.local_files_only,
                ).cuda()
                self.vision_model = Darknet(self.args.darknet_model_file, 224).cuda()

                new_state = torch.load(self.args.darknet_weight_file)
                state = self.vision_model.state_dict()
                model_keys = set(state.keys())
                state_dict = {k: v for k, v in new_state['model'].items() if k in model_keys}
                state.update(state_dict)
                self.vision_model.load_state_dict(state)

                self.lang_model.eval()
                self.vision_model.eval()
                for p in self.lang_model.parameters():
                    p.requires_grad = False
                for p in self.vision_model.parameters():
                    p.requires_grad = False
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                "/mnt/nfs/dell_15td/avdn/Aerial-Vision-and-Dialog-Navigation-origin/Aerial-Vision-and-Dialog-Navigation-origin/datasets/bert-base-uncased",
                local_files_only=True,
            )
            self.lang_model = CustomBERTModel(
                model_name_or_path=self.args.bert_model_path,
                local_files_only=self.args.local_files_only,
            ).cuda()

            self.img_tensor = transforms.ToTensor()
            # self.vit = feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
            # self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').cuda()

            # self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device='cuda')
            # self.feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/deit-tiny-patch16-224')
            self.vision_model = Darknet(self.args.darknet_model_file, 224).cuda()

            new_state = torch.load(self.args.darknet_weight_file)
            state = self.vision_model.state_dict()
            model_keys = set(state.keys())
            state_dict = {k: v for k, v in new_state['model'].items() if k in model_keys}
            state.update(state_dict)
            self.vision_model.load_state_dict(state)


        # create the et model
        self.vln_model = ET(self.args).cuda()
        
        # self.vln_model = ViT_LSTM(
        #     self.args, 
        #     self.vision_model).cuda()
        
        # optimizer        
        assert args.optim in ("adam", "adamW")
        OptimizerClass = torch.optim.Adam if args.optim == "adam" else torch.optim.AdamW
        self.et_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.vln_model.parameters()), lr=args.lr)
        if self.use_qwen_vl:
            self.qwen_optimizer = OptimizerClass(
                [
                    {"params": self.qwen_adapter.text_proj.parameters(), "lr": self.args.lr},
                    {"params": self.qwen_adapter.text_cls_proj.parameters(), "lr": self.args.lr},
                    {"params": self.qwen_adapter.et_vis_proj.parameters(), "lr": self.args.lr},
                    {"params": self.qwen_adapter.qwen_model.parameters(), "lr": self.args.qwen_lr},
                ]
            )
            self.lang_model_optimizer = None
            self.vision_model_optimizer = None
            self.optimizers = (self.et_optimizer, self.qwen_optimizer)
        else:
            self.lang_model_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.lang_model.parameters()), lr=self.args.lr)
            self.vision_model_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.vision_model.parameters()), lr=self.args.lr)
            self.optimizers = (self.et_optimizer, self.lang_model_optimizer, self.vision_model_optimizer)

        #         # Optimizers
        # if self.args.optim == 'rms':
        #     optimizer = torch.optim.RMSprop
        # elif self.args.optim == 'adam':
        #     optimizer = torch.optim.Adam
        # elif self.args.optim == 'adamW':
        #     optimizer = torch.optim.AdamW
        # elif self.args.optim == 'sgd':
        #     optimizer = torch.optim.SGD
        # else:
        #     assert False
        # if self.default_gpu:
        #     print('Optimizer: %s' % self.args.optim)

        # self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        # self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        # self.lang_model_optimizer = optimizer(filter(lambda p: p.requires_grad, self.lang_model.parameters()), lr=self.args.lr)
        # self.vln_model_optimizer = optimizer(filter(lambda p: p.requires_grad, self.vln_model.parameters()), lr=self.args.lr)
        # self.optimizers = (self.lang_model_optimizer, self.vln_model_optimizer)
        
        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, size_average=False)
        self.progress_regression = nn.MSELoss(reduction='sum')
        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
        self.writer = None
        self.global_step = 0

    def get_results(self):
        
        return self.results

    def test(self, loader, env_name = 'no_name_provided', feedback='student', not_in_train = False, **kwargs):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        self.env_name = env_name

        self.vln_model.eval()
        if self.use_qwen_vl:
            self.qwen_adapter.eval()
        else:
            self.lang_model.eval()
            self.vision_model.eval()

        self.losses = []
        self.results = {}
        self.loss = 0
        for l in loader:
            for traj in self.rollout(not_in_train=True, **kwargs): # loop for #batch times
                self.loss = 0
                self.results[traj['instr_id']] = traj

    def train(self, loader, n_epochs, feedback='student', nss_w_weighting = 1, **kwargs):
        ''' Train for a given number of epochs '''
        self.feedback = feedback

        def _format_elapsed(seconds):
            seconds = int(seconds)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"

        record_file = kwargs.get("record_file")
        if record_file:
            write_to_record_file("enter train loop", record_file)
        else:
            print("enter train loop")
            sys.stdout.flush()
        train_start = time.time()

        if self.use_qwen_vl:
            self.qwen_adapter.train()
        else:
            self.lang_model.train()
            self.vision_model.train()
        self.vln_model.train()

        self.losses = []
        for epoch in range(1, n_epochs + 1):
            if self.use_qwen_vl:
                if epoch <= self.args.qwen_freeze_epochs:
                    self.qwen_adapter.set_qwen_trainable(freeze=True)
                else:
                    self.qwen_adapter.set_qwen_trainable(freeze=False, unfreeze_last_n=self.args.qwen_unfreeze_last_layers)
            
            for batch_idx, l in enumerate(tqdm(loader)):
                batch_msg = f"batch_start epoch={epoch} batch={batch_idx} time={_format_elapsed(time.time() - train_start)}"
                if record_file:
                    write_to_record_file(batch_msg, record_file)
                else:
                    print(batch_msg)
                    sys.stdout.flush()
                iter_start = time.time()
                # train_loop_start_time = time.time()
                if self.use_qwen_vl:
                    self.qwen_optimizer.zero_grad()
                else:
                    self.lang_model_optimizer.zero_grad()
                    self.vision_model_optimizer.zero_grad()
                self.et_optimizer.zero_grad()
                self.loss = 0
                
                if feedback == 'teacher':
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.teacher_weight, train_rl=False, nss_w = self.args.nss_w*nss_w_weighting, **kwargs)
                elif feedback == 'student':  # agents in teacher and student separately
                    
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, nss_w = 0, **kwargs)#self.args.nss_w*nss_w_weighting, **kwargs)
                    # if epoch_train > 10000:
                    self.feedback = 'student'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, nss_w = self.args.nss_w*nss_w_weighting, **kwargs)
                else:
                    assert False

                # print("--- One rollout takes %s seconds ---" % (time.time() - train_loop_start_time))
                
                #print(self.rank, epoch, self.loss)
                # torch.autograd.set_detect_anomaly(True)
                

                self.loss.backward()

                if self.writer is not None and self.args.tb_log_every > 0:
                    if self.global_step % self.args.tb_log_every == 0:
                        self.writer.add_scalar("loss/step", float(self.loss), self.global_step)

                torch.nn.utils.clip_grad_norm_(self.vln_model.parameters(), 40.)

                if self.use_qwen_vl:
                    self.qwen_optimizer.step()
                else:
                    self.lang_model_optimizer.step()
                    self.vision_model_optimizer.step()
                self.et_optimizer.step()
                self.global_step += 1
                iter_time = time.time() - iter_start
                elapsed = time.time() - train_start
                iter_msg = f"{self.global_step}it [{_format_elapsed(elapsed)}, {iter_time:.2f}s/it]"
                if record_file:
                    write_to_record_file(iter_msg, record_file)
                else:
                    print(iter_msg)
                    sys.stdout.flush()
                # print("---------- One iter takes %s seconds ---" % (time.time() - train_loop_start_time))

            print_progress(epoch, n_epochs, prefix='Progress:', suffix='Complete', bar_length=50)

    def NSS(self, sal,fix):
        m = torch.mean(sal.view(-1,224*224),1).view(-1,1,1)
        std = torch.std(sal.view(-1,224*224),1).view(-1,1,1)
        if self.args.nss_r == 0:
            n_sal = (sal-m)/std
        elif self.args.nss_r == 1:
            n_sal = (sal-m)/std/2 +1
        elif self.args.nss_r == -1:
            n_sal = (sal-m)/std/2 -1

        s_fix = torch.sum(fix.view(-1,224*224),1)+0.001
        ns = n_sal*fix
        s_ns = torch.sum(ns.view(-1,224*224),1)
        nss = torch.mean(s_ns/s_fix)
        return -nss

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    

    

    # rotate first and then move forward;
    # return the unchanged if hit the map edge
    def move_view_corners(self, corners, angle, distance, altitude, gps_botm_left, gps_top_right, input_current_direction = None): 
        def move_view_corner_forward(cs, change): # corners => cs
            new_cs = np.zeros((4,2))
            new_cs[0] = cs[0] + (cs[0] - cs[3])/ np.linalg.norm((cs[3] - cs[0])) * change

            new_cs[1] = cs[1] + (cs[1] - cs[2])/ np.linalg.norm((cs[2] - cs[1])) * change

            new_cs[2] = cs[2] + (cs[1] - cs[2])/ np.linalg.norm((cs[2] - cs[1])) * change

            new_cs[3] = cs[3] + (cs[0] - cs[3])/ np.linalg.norm((cs[3] - cs[0])) * change

            return new_cs
        def rotation_anticlock(theta, p):
            M = np.array([[np.cos(theta/180*3.14159), np.sin(theta/180*3.14159)], 
                          [-np.sin(theta/180*3.14159), np.cos(theta/180*3.14159)]])
            return np.matmul(M, np.array([p[0], p[1]]))
        def change_corner(cs, change): # corners = cs
            new_cs = np.zeros((4,2))
            new_cs[0] = cs[0] + (cs[0] - cs[1])/ np.linalg.norm((cs[1] - cs[0])) * change
            new_cs[0] += (cs[0] - cs[3])/ np.linalg.norm((cs[3] - cs[0])) * change

            new_cs[1] = cs[1] + (cs[1] - cs[0])/ np.linalg.norm((cs[1] - cs[0])) * change
            new_cs[1] += (cs[1] - cs[2])/ np.linalg.norm((cs[2] - cs[1])) * change

            new_cs[2] = cs[2] + (cs[2] - cs[3])/ np.linalg.norm((cs[2] - cs[3])) * change
            new_cs[2] += (cs[2] - cs[1])/ np.linalg.norm((cs[2] - cs[1])) * change

            new_cs[3] = cs[3] + (cs[3] - cs[2])/ np.linalg.norm((cs[2] - cs[3])) * change
            new_cs[3] += (cs[3] - cs[0])/ np.linalg.norm((cs[3] - cs[0])) * change

            return new_cs
        
        current_direction = round(get_direction(np.mean(corners, axis=0),(corners[0] + corners[1])/2)) % 360
        if input_current_direction != None and abs(input_current_direction - current_direction) >2:
            print('warning, currencting the view area by: +', input_current_direction - current_direction)
            angle += input_current_direction        
        # -------- Zoom --------
        current_view_area_edge_length = np.linalg.norm((corners[1]) - corners[0])*11.13*1e4
        # print('step_to_zoom: ',altitude*400, current_view_area_edge_length)
        # print(corners)
        step_change_of_view_zoom = 0.5*(altitude - current_view_area_edge_length)/11.13/1e4
        _new_corners = change_corner(
            corners,
            step_change_of_view_zoom
        )
        # print(_new_corners)
        
        new_corners = []
        for i in _new_corners:

            if i[0]>gps_botm_left[0] and i[0] < gps_top_right[0] and i[1]>gps_botm_left[1] and i[1]<gps_top_right[1]:
                new_corners.append(i)
            else:
                break
        if len(new_corners) != 4:
            return np.array(corners), current_direction
        corners = new_corners
    
        # -------- Rotate --------
    
        # print(angle)
        mean_im_coords = np.mean(corners, axis=0)
        _corners = [
            corners[0] - mean_im_coords,
            corners[1] - mean_im_coords,
            corners[2] - mean_im_coords,
            corners[3] - mean_im_coords
        ]  # counter clock wise
  
        rotated_corners = []
        for i in range(4):
            rotated_point = mean_im_coords + rotation_anticlock(-angle, _corners[i])
            if rotated_point[0]>gps_botm_left[0] and rotated_point[0] < gps_top_right[0] and \
                rotated_point[1]>gps_botm_left[1] and rotated_point[1]<gps_top_right[1]:
                rotated_corners.append(rotated_point)
            else:
                break
        if len(rotated_corners) != 4:
            return np.array(corners), current_direction

        # -------- Move --------

        
        step_change_of_view_move = distance
        _new_corners = move_view_corner_forward(
                np.array(rotated_corners),
                step_change_of_view_move)


        new_corners = []
        for i in _new_corners:

            if i[0]>gps_botm_left[0] and i[0] < gps_top_right[0] and i[1]>gps_botm_left[1] and i[1]<gps_top_right[1]:
                new_corners.append(i)
            else:
                break
        if len(new_corners) != 4:
            return np.array(rotated_corners), (current_direction + angle) % 360
        else:
            return np.array(new_corners), (current_direction + angle) % 360

    def teacher_action(self, obs, ended, corners, directions):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """

                
        teacher_a = [['0','0'] for x in range(len(obs))]
        progress = np.zeros((len(obs),1), dtype=np.float32)
        for i in range(len(obs)):
            
            current_pos = np.mean(corners[i], axis = 0)

            # -------- calculate the progress (iou) --------   
            iou = compute_iou(corners[i], obs[i]['gt_path_corners'][-1])

            progress[i] = np.float32(iou)


            
            # -------- find teacher altitude --------
            
            min_dis = 1000
            for j in range(len(obs[i]['gt_path_corners'])-1, -1, -1):
                gt_pos = np.mean(obs[i]['gt_path_corners'][j], axis = 0)
                dis_to_current = np.linalg.norm(gt_pos - current_pos)
                if dis_to_current+0.00001<min_dis: # 0.00001 is in case there are two gt_path_corner are the same
                    min_dis = dis_to_current
                    closest_step_index = j
            teacher_a[i][1] = float((np.linalg.norm(obs[i]['gt_path_corners'][closest_step_index][0] -\
                                obs[i]['gt_path_corners'][closest_step_index][1])\
                                *11.13*1e4 -40 )/ (400-40))
            if ended[i] or progress[i] > 0.5:
                teacher_a[i][0] = np.array([0,0], dtype=np.float32)
                continue                          
            # -------- find teacher next_pos --------
            
                    
            goal_corner_center = np.mean(obs[i]['gt_path_corners'][-1], axis = 0)
            polygon = corners[i]
            shapely_poly = shapely.geometry.Polygon(polygon)
            # in teacher forcing learning, the trajectory will be followed step by step
            if self.feedback == 'student': 
                target_point_index = -1
                line = [current_pos]+[np.mean(obs[i]['gt_path_corners'][target_point_index], axis = 0)]
                shapely_line = shapely.geometry.LineString(line)
                intersection_line = list(shapely_poly.intersection(shapely_line).coords)
            else:
                line = [np.mean(obs[i]['gt_path_corners'][j], axis = 0) for j in range(len(obs[i]['gt_path_corners']))]
                shapely_line = shapely.geometry.LineString(line)
                intersection_geom = shapely_poly.intersection(shapely_line)
                if isinstance(intersection_geom, shapely.geometry.linestring.LineString):
                    intersection_line = list(intersection_geom.coords)
                elif isinstance(intersection_geom, shapely.geometry.multilinestring.MultiLineString):
                    intersection_line = []
                    for line_string in intersection_geom.geoms:
                        intersection_line += list(line_string.coords)
                elif isinstance(intersection_geom, shapely.geometry.collection.GeometryCollection):
                    intersection_line = []
                    for geom in intersection_geom.geoms:
                        if isinstance(geom, shapely.geometry.linestring.LineString):
                            intersection_line += list(geom.coords)
                else:
                    intersection_line = []
            

                if intersection_line == []:
                    print(line, closest_step_index)
                    target_point_index = -1
                    line = [current_pos]+[np.mean(obs[i]['gt_path_corners'][target_point_index], axis = 0)]
                    shapely_line = shapely.geometry.LineString(line)
                    intersection_line = list(shapely_poly.intersection(shapely_line).coords)
                
                
            if intersection_line == []:
                print(line, closest_step_index)
            
            min_distance = 1
            for x in intersection_line:
                x = np.array(x)
                _distance = np.linalg.norm(x- goal_corner_center) 
                if _distance < min_distance:
                    min_distance = _distance
                    teacher_a[i][0] = x
                
            # # in student exploring, only targeting the final destination
            # else:
            #     dist = cv2.pointPolygonTest(
            #     cv2.UMat(np.array(( np.array(corners[i]) - goal_corner_center)*100000000, dtype=np.int32)),
            #     (np.array([0,0], dtype=np.float32)), 
            #     True)# >0 means inside the Poly

            #     if dist >0:
            #         teacher_a[i][0] = goal_corner_center
            #     else:
            #         polygon = corners[i]
            #         shapely_poly = shapely.geometry.Polygon(polygon)

            #         line = [current_pos, goal_corner_center]
            #         shapely_line = shapely.geometry.LineString(line)

            #         intersection_line = list(shapely_poly.intersection(shapely_line).coords)
            #         teacher_a[i][0] = np.array(intersection_line[0])

            _net_next_pos = 1e5*(teacher_a[i][0] - current_pos)
            _net_y = np.round(1e5*((corners[i][0] + corners[i][1])/2 - current_pos)).astype(int)
            _net_x = np.round(1e5*((corners[i][1] + corners[i][2])/2 - current_pos)).astype(int)

            A = np.asarray([[_net_x[0], _net_y[0]], [_net_x[1], _net_y[1]]])
            b = np.asarray([_net_next_pos[0], _net_next_pos[1]]).reshape(2, 1)
            r = np.linalg.solve(A,b)

            gt_next_pos_ratio = [r[0,0], r[1,0]]
            
            if max(gt_next_pos_ratio)>1.1:
                print(teacher_a[i][0])
            
  
            max_of_gt_next_pos_ratio= max(abs(gt_next_pos_ratio[0]), abs(gt_next_pos_ratio[1]), 1) # in [-1,1]
            gt_next_pos_ratio[0] /= max_of_gt_next_pos_ratio
            gt_next_pos_ratio[1] /= max_of_gt_next_pos_ratio

            
            teacher_a[i][0] = np.array(gt_next_pos_ratio, dtype=np.float32)

            
            
        return teacher_a, progress
    def gps_to_img_coords(self, gps, gps_botm_left, gps_top_right, lat_ratio):
        return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), int(round((gps_top_right[0] - gps[0]) / lat_ratio))


    def rollout(self, train_ml=None, not_in_train = False, nss_w = 0, **kwargs):

        # rollout_start_time = time.time()

        obs = self.env._get_obs(t=0)
        batch_size = len(obs)


        # Language input
        distill_lang_loss = 0.0
        if self.use_qwen_vl:
            base_lang_inputs = []
            for i, ob in enumerate(obs):
                if self.args.vision_only:
                    base_lang_inputs.append('')
                else:
                    base_lang_inputs.append(ob['instructions'])
            text_tokens, text_cls = self.qwen_adapter.encode_text(base_lang_inputs)
            lang_features, linear_cls = self.qwen_adapter.project_text(text_tokens, text_cls)
            if not self.args.train_val_on_full:
                dialog_inputs = []
                for i, ob in enumerate(obs):
                    dialog_inputs.append(ob['pre_dialogs'] + ob['instructions'])
                text_tokens, text_cls = self.qwen_adapter.encode_text(dialog_inputs)
                _, linear_cls = self.qwen_adapter.project_text(text_tokens, text_cls)
            else:
                dialog_inputs = base_lang_inputs
            lang_inputs = dialog_inputs

            if self.args.use_feature_distill:
                with torch.no_grad():
                    encoding = self.tokenizer(base_lang_inputs, padding=True, return_tensors="pt")
                    input_ids = encoding['input_ids'].cuda()
                    attention_mask = encoding['attention_mask'].cuda()
                    teacher_lang, _, _ = self.lang_model(input_ids, attention_mask)

                    encoding = self.tokenizer(dialog_inputs, padding=True, return_tensors="pt")
                    input_ids = encoding['input_ids'].cuda()
                    attention_mask = encoding['attention_mask'].cuda()
                    _, teacher_cls, _ = self.lang_model(input_ids, attention_mask)
                distill_lang_loss = (
                    self.args.distill_lang_w * F.mse_loss(lang_features, teacher_lang)
                    + self.args.distill_lang_cls_w * F.mse_loss(linear_cls, teacher_cls)
                )
        else:
            lang_inputs = []
            for i, ob in enumerate(obs):
                if self.args.vision_only:
                    lang_inputs.append('')
                else:
                    lang_inputs.append(ob['instructions'])
            encoding = self.tokenizer(lang_inputs,  padding=True, return_tensors="pt")
            input_ids = encoding['input_ids'].cuda()
            attention_mask = encoding['attention_mask'].cuda()
            lang_features, linear_cls, cls_hidden = self.lang_model(input_ids, attention_mask)
            if not self.args.train_val_on_full:
                lang_inputs = []        
                for i, ob in enumerate(obs):
                    lang_inputs.append(ob['pre_dialogs'] + ob['instructions'])
                encoding = self.tokenizer(lang_inputs,  padding=True, return_tensors="pt")
                input_ids = encoding['input_ids'].cuda()
                attention_mask = encoding['attention_mask'].cuda()
                _, linear_cls, cls_hidden = self.lang_model(input_ids, attention_mask)

        # lang_features --> 768
        # linear_cls --> 49 (used to attend to img features)
        # c_0 = cls_hidden
        
        

        # print(lang_features.size()) # batch_size*sequence_length*768



        # Record starting points of the current batch
        current_view_corners = [np.array(ob['gt_path_corners'][0]) for ob in obs]
        current_directions = [np.array(ob['starting_angle']) for ob in obs]
        direction_t = torch.from_numpy(np.array(current_directions))
        traj = [defaultdict(list) for ob in obs]
        for i,ob in enumerate(obs):
            traj[i]['instr_id']= ob['map_name'] + '__' + ob['route_index']
            rounds = lang_inputs[i].split('[QUE]')
            remove = 0
            for r in rounds:
                if 'Yes' in r[0:5]:
                    remove += 1
            traj[i]['num_dia'] = len(rounds) - remove
            traj[i]['path_corners']= [(np.array(ob['gt_path_corners'][0]), ob['starting_angle'])]
            traj[i]['gt_path_corners']= ob['gt_path_corners']

        # Initialization the finishing status
        ended = np.array([False] * batch_size)

        # Init the logs
        ml_loss = 0.

        input = {
            'directions' : torch.zeros((batch_size, 0, 2)).cuda(),
            'frames': torch.zeros(batch_size, 0, 512, 49).cuda(),
            'lenths': [0 for _ in range(batch_size)],
            'lang': lang_features,
            'lang_cls': linear_cls,
        }
        # print("- initialize rollout takes %s seconds ---" % (time.time() - rollout_start_time))
        
        # rollingout_action_start_time = time.time()
        # for t in range(self.args.max_action_len):
        for t in range(self.args.max_action_len):
            # print("- action rollingout takes %s seconds ---" % (time.time() - rollingout_action_start_time))
            # rollingout_action_start_time = time.time()
            distill_frame_loss = 0.0
            if self.use_qwen_vl:
                qwen_images = []
                for i in range(len(obs)):
                    qwen_images.append(obs[i]['current_view'][:, :, ::-1].copy())
                vision_tokens = self.qwen_adapter.encode_image(qwen_images)
                image_size = (qwen_images[0].shape[1], qwen_images[0].shape[0])
                im_feature = self.qwen_adapter.project_vision_for_et(vision_tokens, image_size=image_size)

                if self.args.use_feature_distill:
                    images = []
                    for i in range(len(obs)):
                        images.append(obs[i]['current_view'].copy())
                    images = np.stack(images)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # W x H x C to C x W x H
                    images = np.ascontiguousarray(images, dtype=np.float32)
                    images -= self.rgb_mean
                    images /= self.rgb_std
                    with torch.no_grad():
                        teacher_im = self.vision_model(torch.from_numpy(images).cuda())
                        teacher_im = teacher_im.view(teacher_im.size(0), teacher_im.size(1), -1)
                    distill_frame_loss = self.args.distill_frame_w * F.mse_loss(im_feature, teacher_im)
            else:
                images = []
                for i in range(len(obs)):
                    images.append(obs[i]['current_view'].copy())
                images = np.stack(images)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # W x H x C to C x W x H
                images = np.ascontiguousarray(images, dtype=np.float32)
                images -= self.rgb_mean
                images /= self.rgb_std
                im_feature = self.vision_model(torch.from_numpy(images).cuda())
                im_feature = im_feature.view(im_feature.size(0), im_feature.size(1), -1)
            
            # pos_tensor = []
            # for i in range(len(obs)):
            #     pos_tensor.append(
            #                     (np.mean(traj[i]['path_corners'][0][0], axis=0) -
            #                     np.mean(current_view_corners[i], axis=0))/0.001
            #                     )
            # pos_tensor = torch.from_numpy(np.array(pos_tensor, dtype=np.float32)).cuda()
                
            
            current_direct = direction_t.view(-1, 1).cuda()
            direction = torch.concat((torch.sin(current_direct/180*3.14159),torch.cos(current_direct/180*3.14159)),axis = 1)

            if self.args.no_direction:
                input['directions'] = torch.hstack((input['directions'], torch.zeros_like(direction.view(-1,1,2))))
            else:
                input['directions'] = torch.hstack((input['directions'], direction.view(-1,1,2)))
            if self.args.language_only:
                input['frames'] = torch.hstack((input['frames'], torch.zeros_like(im_feature.view(-1,1, 512,49))))
            else:
                input['frames'] = torch.hstack((input['frames'], im_feature.view(-1,1, 512,49)))
            
            # input['directions'] = direction.view(-1,1,2)
            # input['frames'] = im_feature.view(-1,1, 512,49)
            
            for i in range(len(obs)):
                if not ended[i]:
                    input['lenths'][i] += 1
            
            output, pred_saliency = self.vln_model(
                directions = input['directions'],
                frames = input['frames'],
                lenths = input['lenths'],
                lang = input['lang'],
                lang_cls = input['lang_cls']
                )
            # print("- model prediction takes %s seconds ---" % (time.time() - rollingout_action_start_time))
            pred_next_pos_ratio = output[:,0:2]
            pred_altitude = output[:,2]
            pred_progress = output[:,3]

            # Predicted progress
            pred_progress_t = pred_progress.cpu().detach().numpy() 

            # Predicted waypoint
            a_t_next_pos_ratio = pred_next_pos_ratio.cpu().detach().numpy()
            for i in range(len(a_t_next_pos_ratio)):
                max_of_a_t_next_pos_i = max(abs(a_t_next_pos_ratio[i][0]), abs(a_t_next_pos_ratio[i][1]), 1)
                a_t_next_pos_ratio[i][0] /= max_of_a_t_next_pos_i
                a_t_next_pos_ratio[i][1] /= max_of_a_t_next_pos_i

            # Predicted altitude
            a_t_altitude = pred_altitude.cpu().detach().numpy() 

            # Clip the prediction to (0,1)
            for i in range(len(a_t_altitude)):
                a_t_altitude[i] = min(1., max(0., a_t_altitude[i]))
            for i in range(len(pred_progress_t)):
                pred_progress_t[i] = min(1., max(0., pred_progress_t[i]))


            # there is no ground truth in unseen_test set
            if not 'test' in self.env_name:
                # Get ground truth
                target, gt_progress = self.teacher_action(obs, ended, current_view_corners, current_directions) # Retrun gt action for every batch that have not reached the end
                # print(t, target, gt_progress)
                
                # Compute loss
                for i in range(len(obs)):
                        cuda_gt_next_pos_ratio = torch.from_numpy(target[i][0]).cuda()
                        ml_loss += self.progress_regression(pred_next_pos_ratio[i,:], cuda_gt_next_pos_ratio)
                        ml_loss += self.progress_regression((torch.atan2(pred_next_pos_ratio[i,0], pred_next_pos_ratio[i,1]+1e-5*np.random.rand(1)[0])  /3.14159 + 2) / 2  %1 ,
                                                            (torch.atan2(cuda_gt_next_pos_ratio[0], cuda_gt_next_pos_ratio[1])  /3.14159 + 2) / 2  %1)
                        ml_loss += self.progress_regression(pred_altitude[i].view(-1), torch.tensor([target[i][1]]).cuda())
                        ml_loss += self.progress_regression(pred_progress[i].view(-1), torch.tensor([gt_progress[i,0]]).cuda())
                        if ml_loss != ml_loss: # debug for nan loss
                            print('0', ml_loss)
                
                # 梯度裁剪：防止梯度爆炸导致数值溢出
                if train_ml is not None:
                    torch.nn.utils.clip_grad_norm_(self.vln_model.parameters(), 5.0)
                    if self.use_qwen_vl:
                        torch.nn.utils.clip_grad_norm_(self.qwen_adapter.parameters(), 5.0)

                if self.use_qwen_vl and self.args.use_feature_distill and train_ml is not None:
                    ml_loss += distill_lang_loss
                    ml_loss += distill_frame_loss
                # Human attention prediction and NSS loss
                if not getattr(self.args, "disable_haa", False):
                    for i in range(len(obs)):
                        pred_saliency_cpu = pred_saliency[i].clip(0,1).cpu().detach().numpy().reshape(224,224,1)
                        gt_saliency = obs[i]['gt_saliency'].reshape(224,224,1)
                        if np.sum(obs[i]['gt_saliency']) > 0:
                            nss_loss= self.NSS(pred_saliency[i], torch.from_numpy(obs[i]['gt_saliency']).cuda())
                            if nss_loss != nss_loss: # debug for nan loss
                                print('1', nss_loss)
                            else:
                                ml_loss += nss_w*nss_loss
                            # in human att evaluation
                            if not_in_train == True and self.feedback == 'teacher':
                                tp = np.sum(pred_saliency_cpu*gt_saliency, dtype=np.float32)
                                if np.sum(pred_saliency_cpu, dtype=np.float32) != 0:
                                    precision = tp/np.sum(pred_saliency_cpu, dtype=np.float32)
                                else:
                                    precision = 0.
                                recall = tp/np.sum(gt_saliency, dtype=np.float32)
                                traj[i]['human_att_performance'].append([precision, recall])
                                traj[i]['nss'].append(nss_loss.item())
                        
                        if self.args.inference and self.feedback == 'teacher':    
                            # cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_'+ obs[i]['route_index']+ '_pred_att_' + str(t) +'.jpg',
                            #         obs[i]['current_view'] * np.repeat(pred_saliency_cpu/np.max(pred_saliency_cpu), 3, axis = 2))
                            # cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_'+ obs[i]['route_index']+ '_gt_att_' + str(t) +'.jpg',
                            #         obs[i]['current_view'] * np.repeat(gt_saliency, 3, axis = 2))

                            cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_'+ obs[i]['route_index']+ '_pred_att_' + str(t) +'.jpg',
                                    cv2.applyColorMap(np.uint8(255*(pred_saliency_cpu/np.max(pred_saliency_cpu))), cv2.COLORMAP_JET))
                            cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_'+ obs[i]['route_index']+ '_gt_att_' + str(t) +'.jpg',
                                    cv2.applyColorMap(np.uint8(255*gt_saliency), cv2.COLORMAP_JET))
                                    
                            cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_'+ obs[i]['route_index']+ '_input_' + str(t) +'.jpg',
                                    obs[i]['current_view'])
                    
                    # else:
                    #     cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'train' + obs[i]['map_name'] + '_'+ obs[i]['route_index']+ '_pred_att_' + str(t) +'.jpg',
                    #             obs[i]['current_view'] * np.repeat(pred_saliency.cpu().detach().numpy().reshape(224,224,1), 3, axis = 2))
                    #     cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'train' + obs[i]['map_name'] + '_'+ obs[i]['route_index']+ '_gt_att_' + str(t) +'.jpg',
                    #             obs[i]['current_view'] * np.repeat(obs[i]['gt_saliency'].reshape(224,224,1), 3, axis = 2))
            

            # Log the trajectory
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['actions'].append([a_t_next_pos_ratio[i], a_t_altitude[i]])
                    if not 'test' in self.env_name:
                        traj[i]['gt_actions'].append(target[i])
                        traj[i]['gt_progress'].append(gt_progress[i].item())
                    traj[i]['progress'].append(pred_progress[i].item())

            if self.feedback == 'teacher':   
                a_t = target
                pred_progress_t = gt_progress
            elif self.feedback == 'student': # student
                a_t = [[a_t_next_pos_ratio[j],a_t_altitude[j]] for j in range(len(obs))]
            else:
                sys.exit('Invalid feedback option')

            # Interact with the simulator with actions
            for i in range(len(obs)):
                if pred_progress_t[i] >0.5 and self.feedback == 'teacher':
                    # Updated 'ended' list and make environment action
                    ended[i] = True
                    continue
                elif pred_progress_t[i] >0.5 and self.feedback == 'student':
                    # Updated 'ended' list and make environment action
                    ended[i] = True
                    continue
                elif t == self.args.max_action_len - 1:
                    ended[i] = True
                    continue
                
                # 数值安全检查：拦截 NaN 防止 round() 函数崩溃
                if np.isnan(a_t[i][0][0]) or np.isnan(a_t[i][0][1]) or np.isnan(a_t[i][1]):
                    # print(f"Warning: NaN detected in action prediction at step {t}, early exit.")
                    ended[i] = True
                    continue

                a_direction = (math.atan2(a_t[i][0][0], a_t[i][0][1]) /3.14159 + 2) / 2  %1
                a_distance = np.linalg.norm(a_t[i][0]) * (np.linalg.norm(current_view_corners[i][0] - current_view_corners[i][1])/2) # * view_coner edge lenth
                a_altitude = a_t[i][1]
                new_current_view_corners, current_directions[i]  = self.move_view_corners(
                    current_view_corners[i], 
                    round(a_direction*360), 
                    a_distance,
                    round(a_altitude * 360) +40,
                    obs[i]['gps_botm_left'],
                    obs[i]['gps_top_right'],
                    current_directions[i])
                # print('end:', ended[i])
                current_view_corners[i] = new_current_view_corners
                
            # Save trajectory output
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path_corners'].append(
                            (current_view_corners[i] , current_directions[i])
                            ) 
            # Update the status
            direction_t = torch.from_numpy(np.array(current_directions))
            obs = self.env._get_obs(corners = current_view_corners, directions = current_directions) # get gt_obs
            # current_view_corners = [np.array(ob['gt_path_corners'][0]) for ob in obs]

            # Early exit if all ended   
            if ended.all(): 
                break
            
            
        # For inference. Visualization is saved.
        if self.args.inference == True and not 'test' in self.env_name :
            for i in range(len(obs)):
                if 1: #obs[i]['map_name'].split('_')[0][-1] == '8': # just visualize some of the data
                    img = self.env.map_batch[obs[i]['map_name']].copy()
 

                    for j in range(len(traj[i]['actions'])):
                        mean_coord = self.gps_to_img_coords(
                            np.mean(traj[i]['path_corners'][j][0], axis = 0), 
                            obs[i]['gps_botm_left'],
                            obs[i]['gps_top_right'],
                            obs[i]['lat_ratio'])
                        mean_coord = np.array(mean_coord, dtype = np.int32)
      
                        a_direction = (math.atan2(traj[i]['actions'][j][0][0], traj[i]['actions'][j][0][1]) /3.14159 + 2) / 2  %1
                        a_distance = np.linalg.norm(traj[i]['actions'][j][0]) * (np.linalg.norm(traj[i]['path_corners'][j][0][0] - traj[i]['path_corners'][j][0][1])/2) # * view_coner edge lenth
                        a_altitude = traj[i]['actions'][j][1]
                        # print('action: ', a_direction, a_altitude)

                        # print(next_coord)
                        
                        # Draw the bounding box for view area.
                        # eg. at time_step = 0, draw the first bounding box for time_step=0
                        cv2.drawContours(img, [np.array(
                        [[self.gps_to_img_coords([traj[i]['path_corners'][j][0][0][0], traj[i]['path_corners'][j][0][0][1]], obs[i]['gps_botm_left'],
                            obs[i]['gps_top_right'],
                            obs[i]['lat_ratio'])], 
                        [self.gps_to_img_coords([traj[i]['path_corners'][j][0][1][0], traj[i]['path_corners'][j][0][1][1]], obs[i]['gps_botm_left'],
                            obs[i]['gps_top_right'],
                            obs[i]['lat_ratio'])], 
                        [self.gps_to_img_coords([traj[i]['path_corners'][j][0][2][0], traj[i]['path_corners'][j][0][2][1]], obs[i]['gps_botm_left'],
                            obs[i]['gps_top_right'],
                            obs[i]['lat_ratio'])],
                        [self.gps_to_img_coords([traj[i]['path_corners'][j][0][3][0], traj[i]['path_corners'][j][0][3][1]], obs[i]['gps_botm_left'],
                            obs[i]['gps_top_right'],
                            obs[i]['lat_ratio'])]])], 0, (255, 255, 255), 1)
                        
                        # Compute the bounding box according to the action taken at time_step=0
                        # but only draw a line connecting the bounding box at time_step=0 and the bounding box at time_step=1
                        next_coord, _ = self.move_view_corners(
                                traj[i]['path_corners'][j][0], 
                                round(a_direction*360), 
                                a_distance,
                                round(a_altitude * 360) +40,
                                obs[i]['gps_botm_left'],
                                obs[i]['gps_top_right'])
                        next_coord = self.gps_to_img_coords(
                            np.mean(next_coord, axis = 0), 
                            obs[i]['gps_botm_left'],
                            obs[i]['gps_top_right'],
                            obs[i]['lat_ratio'])
                        cv2.line(img, mean_coord, next_coord, (255, 0, 255), 4)
                        cv2.circle(img, mean_coord, color=(255,255,255), thickness=2, radius=2) 
                    
                        # Draw the ground truth action. There are two situation: 
                        # the gt_action shows the current view area has already reached the destination
                        if type(traj[i]['gt_actions'][j][0]) == type(-100):
                            cv2.putText(img, str(j)+': ['+str(traj[i]['actions'][j][0][0])[:4]+','+ str(traj[i]['actions'][j][0][1])[:4] + '; ' +\
                                         str(traj[i]['gt_actions'][j][0]) + '] : ' + str(traj[i]['progress'][j])[:4]+', '+str(traj[i]['gt_progress'][j])[:4], 
                                    self.gps_to_img_coords([traj[i]['path_corners'][j][0][0][0], traj[i]['path_corners'][j][0][0][1]], obs[i]['gps_botm_left'],
                                        obs[i]['gps_top_right'],
                                        obs[i]['lat_ratio']), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                        # the gt_action shows the destination has not reached.
                        else:
                            cv2.putText(img, str(j)+': ['+str(traj[i]['actions'][j][0][0])[:4]+','+ str(traj[i]['actions'][j][0][1])[:4] + '; ' +\
                                         str(traj[i]['gt_actions'][j][0][0])[:4] +','+ str(traj[i]['gt_actions'][j][0][1])[:4] + '] : ' + str(traj[i]['progress'][j])[:4]+', '+str(traj[i]['gt_progress'][j])[:4], 
                                    self.gps_to_img_coords([traj[i]['path_corners'][j][0][0][0], traj[i]['path_corners'][j][0][0][1]], obs[i]['gps_botm_left'],
                                        obs[i]['gps_top_right'],
                                        obs[i]['lat_ratio']), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                            a_direction = (math.atan2(traj[i]['gt_actions'][j][0][0], traj[i]['gt_actions'][j][0][1]) /3.14159 + 2) / 2  %1
                            a_distance = np.linalg.norm(traj[i]['gt_actions'][j][0]) * (np.linalg.norm(traj[i]['path_corners'][j][0][0] - traj[i]['path_corners'][j][0][1])/2) # * view_coner edge lenth
                            a_altitude = traj[i]['gt_actions'][j][1]
                            # print('gt_action: ', a_direction, a_altitude)

                            next_coord, _ = self.move_view_corners(
                                    traj[i]['path_corners'][j][0], 
                                    round(a_direction*360), 
                                    a_distance,
                                    round(a_altitude * 360) +40,
                                    obs[i]['gps_botm_left'],
                                    obs[i]['gps_top_right'])
                            # print(next_coord)
                            next_coord = self.gps_to_img_coords(
                                np.mean(next_coord, axis = 0),
                                obs[i]['gps_botm_left'],
                                obs[i]['gps_top_right'],
                                obs[i]['lat_ratio'])
                            cv2.line(img, mean_coord, np.array((mean_coord*2/3+np.array(next_coord)/3), dtype=np.int32), (0, 255, 0), 4)
                        
                        

                    cv2.putText(img, obs[i]['instructions'], 
                                        (50,50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    if not_in_train == True:
                        cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'val' + obs[i]['map_name'] + '_'+ obs[i]['route_index']+'.jpg',
                    img)
                
                    # else:
                    #     cv2.imwrite(self.args.pred_dir + '/debug_images/' + self.env_name + 'train' + obs[i]['map_name'] + '_'+ obs[i]['route_index']+'.jpg',
                    # img)


        
        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.
        # print('[3]')
        # debug_memory()
        # print()
        return traj




    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_model", self.vln_model, self.et_optimizer)]
        if self.use_qwen_vl:
            all_tuple.append(("qwen_adapter", self.qwen_adapter, self.qwen_optimizer))
        else:
            all_tuple.append(("lang_model", self.lang_model, self.lang_model_optimizer))
            all_tuple.append(("vision_model", self.vision_model, self.vision_model_optimizer))
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys == load_keys:
                print("NOTICE: LOADing ALL KEYS IN THE ", name)
                state_dict = states[name]['state_dict']
            else:
                print("NOTICE: DIFFERENT KEYS IN THE ", name)
                # if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                #     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                state_dict = {k: v for k, v in states[name]['state_dict'].items() if k in model_keys}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
            def count_parameters(mo): return sum(p.numel() for p in mo.parameters() if p.requires_grad)
            print('Model parameters: ', count_parameters(model))
        all_tuple = [("vln_model", self.vln_model, self.et_optimizer)]
        if self.use_qwen_vl:
            all_tuple.append(("qwen_adapter", self.qwen_adapter, self.qwen_optimizer))
        else:
            all_tuple.append(("lang_model", self.lang_model, self.lang_model_optimizer))
            all_tuple.append(("vision_model", self.vision_model, self.vision_model_optimizer))
        for param in all_tuple:
            recover_state(*param)
        return states['vln_model']['epoch'] - 1




