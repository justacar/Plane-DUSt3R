import sys
import os.path as path
import argparse
HERE_PATH = path.normpath(path.dirname(__file__))
DUST3R_REPO_PATH = path.normpath(path.join(HERE_PATH, 'MASt3R','mast3r'))
if path.isdir(DUST3R_REPO_PATH):
    # workaround for sibling import
    sys.path.insert(0, path.join(HERE_PATH,  'MASt3R'))

import os
import numpy as np
import json
import traceback
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from easydict import EasyDict
import yaml
import torch

from MASt3R.dust3r_extract import dust3r_extract
from dust3r.model import AsymmetricCroCo3DStereo
from NonCuboidRoom.plane_detection import extract_plane
from NonCuboidRoom.noncuboid.models import Detector
from easydict import EasyDict
from plane_merge_planedust3r import plane_merge
from metric import metric_geodust3r
import yaml
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
plt.switch_backend('agg')


metric_flag = False


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PlaneDust3r')
    parser.add_argument('--dust3r_model', type=str, required=True,
                        help='Path to the Dust3r model checkpoint')
    parser.add_argument('--noncuboid_model', type=str, required=True,
                        help='Path to the NonCuboid model checkpoint')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path containing the scenes to process')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the results')
    parser.add_argument('--save_flag', type=bool, default=True,
                        help='Save the results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Replace hardcoded paths with arguments
    dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(args.dust3r_model).to(args.device)

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
    image_count = 0
    avg_results = np.zeros(5)
    room_count = 0
    whole_precision, whole_recall = 0, 0
    save = False

    view_results = {i: {"avg_results": np.zeros(5), 
                       "image_count": 0,
                       "room_count": 0,
                       "actual_room_count": 0,
                       "precision": 0,
                       "recall": 0} for i in range(5)}

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
                view_results[len(image_list)-1]["actual_room_count"] += 1
                result_dir = os.path.join(save_path,scene_number, room_id)
                os.makedirs(result_dir, exist_ok=True)
              
                try:
                    if os.path.exists(os.path.join(result_dir, 'dust3r_output.npz')):
                        dust3r_output = np.load(os.path.join(result_dir, 'dust3r_output.npz'))
                    else:
                        dust3r_output = dust3r_extract(image_list, dust3r_model, save=save, filename=os.path.join(result_dir, 'dust3r_output.npz'), metric = metric_flag)
                    
                    if os.path.exists(os.path.join(result_dir, 'plane_detection.json')):
                        with open(os.path.join(result_dir, "plane_detection.json"), 'r') as f:
                            plane_detection = json.load(f)
                    else:
                        plane_detection = extract_plane(image_list, noncuboid_model, cfg, save=save, filename=str(os.path.join(result_dir, 'plane_detection.json')))
                    
                    if os.path.exists(os.path.join(result_dir, 'node_data.json')):
                        with open(os.path.join(result_dir, "node_data.json"), 'r') as f:
                            node_info = json.load(f)
                    else:
                        node_info = plane_merge(dust3r_output, plane_detection, save=save, filedir=result_dir, metric=metric_flag)
                    metric_results,precision,recall = metric_geodust3r(plane_detection, dust3r_output, node_info, image_list, save=save, filedir=result_dir, metric=metric_flag)
                    for result in metric_results:
                        view_results[len(image_list)-1]["avg_results"] += result
                        view_results[len(image_list)-1]["image_count"] += 1
                        avg_results += result
                        image_count += 1
                    
                    room_count += 1
                    view_results[len(image_list)-1]["precision"] += precision
                    view_results[len(image_list)-1]["recall"] += recall
                    view_results[len(image_list)-1]["room_count"] += 1
                    whole_precision += precision
                    whole_recall += recall
               
                    LOG.info(" ".join([f"{result:.4f}" for result in (avg_results / image_count)[:-1]])+f" precision: {whole_precision/room_count:.4f}, recall: {whole_recall/room_count:.4f}")
                except Exception as e:
                    continue

    whole_precision /= room_count
    whole_recall /= room_count

   