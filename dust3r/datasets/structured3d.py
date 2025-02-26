# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Co3d_v2
# dataset at https://github.com/facebookresearch/co3d - Creative Commons Attribution-NonCommercial 4.0 International
# See datasets_preprocess/preprocess_co3d.py
# --------------------------------------------------------
import os.path as osp
import json
import itertools
from collections import deque

import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class Structured3d(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.is_metric_scale = True
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg

        # load all scenes
        with open(osp.join(self.ROOT, f'{self.split}_set.json'), 'r') as f:
            self.scene_list = json.load(f)
            
 
        # self.invalidate = {scene: {} for scene in self.scene_list}

    def __len__(self):
        return len(self.scene_list)

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        scene_data = self.scene_list[idx]
        sceneID = scene_data["sceneID"]
        roomID = scene_data["roomID"]
        imgs_idxs = [scene_data["positionID1"], scene_data["positionID2"]]

        # if resolution not in self.invalidate[obj, instance]:  # flag invalid images
        #     self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            # if self.invalidate[obj, instance][resolution][im_idx]:
            #     # search for a valid image
            #     random_direction = 2 * rng.choice(2) - 1
            #     for offset in range(1, len(image_pool)):
            #         tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
            #         if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
            #             im_idx = tentative_im_idx
            #             break


            impath = osp.join(self.ROOT, sceneID, "2D_rendering", roomID, 'perspective','full', im_idx, 'rgb_rawlight.png')

            # load camera params
            input_metadata = np.load(impath.replace('rgb_rawlight.png', 'camera_pose.npz'))
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = cv2.imread(impath.replace('rgb_rawlight.png','plane_depth.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0 #TODO: check if this is correct
            #TODO following for original depth , not plane depth
            # fx = 762  
            # fy = 762 
            # cx = 640  
            # cy = 360  
            # height, width = depthmap.shape
            # u, v = np.meshgrid(np.arange(width), np.arange(height))

            # # Convert pixel coordinates to camera coordinates
            # x = (u - cx) / fx
            # y = (v - cy) / fy

            # # Calculate z from the Euclidean distance
            # # depth_map = np.sqrt(x^2 + y^2 + z^2)
            # # Rearrange to solve for z: z = sqrt(depth_map^2 - x^2 - y^2)
            # depthmap = depthmap / np.sqrt(1 + x**2 + y**2).astype(np.float32)
            
            
            
            # if mask_bg:
            #     # load object mask
            #     maskpath = osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')
            #     maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
            #     maskmap = (maskmap / 255.0) > 0.1

            #     # update the depthmap with mask
            #     depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            # num_valid = (depthmap > 0.0).sum()
            # if num_valid == 0:
            #     # problem, invalidate image and retry
            #     self.invalidate[obj, instance][resolution][im_idx] = True
            #     imgs_idxs.append(im_idx)
            #     continue

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='Structured3D',
                label=osp.join(sceneID, roomID, im_idx),
                instance=osp.split(impath)[1],
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size,pts3d_to_trimesh,cat_meshes
    from dust3r.utils.image import rgb
    import trimesh
    dataset = Structured3d(split='train', ROOT="", resolution=512, aug_crop=16)
    dataset_test = Structured3d(split='test', ROOT="", resolution=512, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        if np.max(views[0]['depthmap']) > 11 or np.max(views[1]['depthmap']) > 11:
            print(np.max(views[0]['depthmap']), np.max(views[1]['depthmap']))
        # viz = SceneViz()
        # poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        # cam_size = max(auto_cam_size(poses), 0.001)
        # scene = trimesh.Scene()
        # meshes = []
        # for view_idx in [0, 1]:
        #     pts3d = views[view_idx]['pts3d']
        #     valid_mask = views[view_idx]['valid_mask']
        #     colors = rgb(views[view_idx]['img'])
        #     # viz.add_pointcloud(pts3d, colors, valid_mask)
        #     # viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
        #     #                focal=views[view_idx]['camera_intrinsics'][0, 0],
        #     #                color=(idx*255, (1 - idx)*255, 0),
        #     #                image=colors,
        #     #                cam_size=cam_size)
        #     meshes.append(pts3d_to_trimesh(rgb(views[view_idx]['img']), views[view_idx]['pts3d'], views[view_idx]['valid_mask']))
        # mesh = trimesh.Trimesh(**cat_meshes(meshes))
        # scene.add_geometry(mesh)
        #     # print(colors[0])
        # # viz.save_scene()
        # scene.export(file_obj="scene.glb")
        



