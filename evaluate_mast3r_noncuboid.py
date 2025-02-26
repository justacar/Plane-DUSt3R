import sys
import os.path as path
import argparse
HERE_PATH = path.normpath(path.dirname(__file__))
DUST3R_REPO_PATH = path.normpath(path.join(HERE_PATH, 'MASt3R','mast3r'))
if path.isdir(DUST3R_REPO_PATH):
    # workaround for sibling import
    sys.path.insert(0, path.join(HERE_PATH,  'MASt3R'))

import os
import torch

from MASt3R.mast3r_extract import mast3r_extract
from NonCuboidRoom.genlayout import extract_layout
from plane_merge_mast3r import plane_merge
from metric import metric_mast3r

from mast3r.model import AsymmetricMASt3R

from NonCuboidRoom.noncuboid.models import Detector
from easydict import EasyDict
import yaml

import logging 
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import traceback
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils import parse_camera_info

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MASt3R')
    parser.add_argument('--mast3r_model', type=str, required=True,
                        help='Path to the MASt3R model checkpoint')
    parser.add_argument('--noncuboid_model', type=str, required=True,
                        help='Path to the NonCuboid model checkpoint') # please MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth from offical repo of MASt3R
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path containing the scenes to process')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the results')
    parser.add_argument('--save_flag', type=bool, default=True,
                        help='Save the results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--gt_flag', type=bool, default=False,
                        help='Use ground truth poses')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Replace hardcoded paths with arguments
    model = AsymmetricMASt3R.from_pretrained(args.mast3r_model).to(args.device)

    noncuboid_model = Detector()
    state_dict = torch.load(args.noncuboid_model, map_location=torch.device(args.device))
    noncuboid_model.load_state_dict(state_dict)
    noncuboid_model = noncuboid_model.to(args.device)
    
    cfg_path = "NonCuboidRoom/cfg.yaml"
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
        cfg = EasyDict(config)

    root_path = args.root_path
    save_path = args.save_path
    gt_flag = args.gt_flag
    
    image_count = 0
    avg_results = np.zeros(5)
    room_count = 0
    whole_precision, whole_recall = 0, 0

    with logging_redirect_tqdm():
        for scene_id in tqdm(sorted(os.listdir(root_path))):
        
            LOG.info(f"Processing scene {scene_id}")
            scene_path = os.path.join(root_path, scene_id)
            scene_number = scene_id.split('_')[1]

            perspective_path = os.path.join(scene_path, "2D_rendering")
            for room_id in os.listdir(perspective_path):
              
                room_path = os.path.join(perspective_path, room_id, 'perspective', 'full')
                position_ids = sorted(os.listdir(room_path))

                image_list = []
                for position_id in position_ids:
                    position_path = os.path.join(room_path, position_id)
                    image_list.append(os.path.join(position_path, 'rgb_rawlight.png'))

                result_dir = os.path.join(save_path,scene_number, room_id)
                os.makedirs(result_dir, exist_ok=True)
              
                try:
                    if gt_flag:
                        poses = []
                        # RT0, _ = parse_camera_info(np.loadtxt(f"{room_path}/0/camera_pose.txt"), 720, 1280)
                        for i in range(len(image_list)):
                            RT, _ = parse_camera_info(np.loadtxt(f"{room_path}/{i}/camera_pose.txt"), 720, 1280)
                            # RT_gt = np.linalg.inv(RT) @ RT0
                            poses.append(RT)
                    else:
                        if os.path.exists(os.path.join(result_dir, 'poses.npy')):
                            poses = np.load(os.path.join(result_dir, 'poses.npy'))
                        else:
                            poses = mast3r_extract(image_list, model,save=True,filename=f"{result_dir}/poses.npy")
       
                    non_cuboid_results, pointmaps = extract_layout(image_list, noncuboid_model, cfg)

                    node_info = plane_merge(pointmaps, poses, non_cuboid_results, save = True, save_dir = result_dir)

                    metric_results,precision,recall = metric_mast3r(non_cuboid_results, poses, node_info,image_list,save = True, filedir = result_dir)   
                    for result in metric_results:
                        avg_results += result
                        image_count += 1
                    
                    room_count += 1
                    whole_precision += precision
                    whole_recall += recall
               
                    LOG.info(" ".join([f"{result:.4f}" for result in (avg_results / image_count)[:-1]])+f" precision: {whole_precision/room_count:.4f}, recall: {whole_recall/room_count:.4f}")
                except Exception as e:
                    continue
                    

    # whole_precision /= room_count
    # whole_recall /= room_count
    # print(whole_precision, whole_recall)
