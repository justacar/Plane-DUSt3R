import os
import sys
HERE_PATH = os.path.normpath(os.path.dirname(__file__))
NONCUBOID_REPO_PATH = os.path.normpath(os.path.join(HERE_PATH, 'noncuboid'))
if os.path.isdir(NONCUBOID_REPO_PATH):
    # workaround for sibling import
    sys.path.insert(0, HERE_PATH)
import numpy as np
import torch
from noncuboid.datasets import  Dust3rDataset
from noncuboid.models import (ConvertLayout, Detector, DisplayLayout, display2Dseg, Loss,
                    Reconstruction, _validate_colormap, post_process)
import yaml
from easydict import EasyDict
import cv2
import matplotlib.pyplot as plt
import json

downsample = 8



def get_scaled_intrinsic(K_original, original_size, new_size):
     # New intrinsic matrix for image size 512x288
    # Scaling factors for focal length and principal point coordinates
    '''
        size(x,y)
    '''
    fx_scale = new_size[0] / original_size[0]
    fy_scale = new_size[1] / original_size[1]
    cx_scale = fx_scale 
    cy_scale = fy_scale
    
    # Adjusted focal lengths and principal points
    fx_new = K_original[0,0] * fx_scale
    fy_new = K_original[1,1] * fy_scale
    cx_new = K_original[0,2] * cx_scale
    cy_new = K_original[1,2] * cy_scale
    
    K_new = np.array([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]], dtype=np.float32)
    return K_new


def get_segmentation(_ups, _downs, pfloor, pceiling, _params_layout):

    _ups = np.array(_ups).astype(np.int32)
    _downs = np.array(_downs).astype(np.int32)
    h,w = 360,640
    segmentation = -1 * np.ones([h, w])  # [0: ceiling, 1: floor, 2...:walls]

   
    
    if len(pceiling)>0: 
        minuy = min(np.min(_ups[:, 1]) - 10, -1)
        cv2.fillPoly(img=segmentation, pts=np.array(
        [[[_ups[0, 0], minuy], *_ups, [_ups[-1, 0], minuy]]]), color=0)
    if len(pfloor)>0: 
        maxdy = max(np.max(_downs[:, 1]) + 10, h+1)
        cv2.fillPoly(img=segmentation, pts=np.array(
            [[[_downs[0, 0], maxdy], *_downs, [_downs[-1, 0], maxdy]]]), color=1)

    assert len(_ups) == len(_params_layout) + 1
    j = -1
    for i in range(len(_ups)-1):
        u0 = _ups[i]
        u1 = _ups[i+1]
        
        if _params_layout[i] is None:
            continue
        d0 = _downs[i]
        d1 = _downs[i+1]

        j = j + 1

        cv2.fillPoly(img=segmentation, pts=np.array(
            [[u0, d0, d1, u1]]), color=j+2)
    return segmentation

def get_depth(segmentation, pfloor, pceiling, pwalls, new_K):
    pwinverdepth = np.ones_like(segmentation) * 1e5
    # pixelwise
    
    depth = np.ones_like(segmentation) * 1e5
    labels = np.unique(segmentation)
    h,w = segmentation.shape
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    xymap = np.stack([xx, yy], axis=2).astype(np.float32)
    ixymap = cv2.resize(xymap, (w, h), interpolation=cv2.INTER_LINEAR)
    ixy1map = np.concatenate([
            ixymap, np.ones_like(ixymap[:, :, :1])], axis=-1).astype(np.float32)
    


    for i, label in enumerate(labels):
        label = int(label)
        if label == -1:
            assert i == 0
            continue
        mask = segmentation == label
        if label == 0:
            param = pceiling
        elif label == 1:
            param = pfloor
        else:
            param = pwalls[label-2]
            if param[3] < 0:
                param[3] = abs(param[3])
                param[0] = -param[0]
                param[2] = -param[2]

        if param is None:
            raise IOError
        else:
            
            n_d = param[:3] / np.clip(param[3], 1e-8, 1e8)  # meter n/d
            n_d = n_d[np.newaxis, np.newaxis, :]
            inverdepth = -1 * np.sum(np.dot(n_d, np.linalg.inv(new_K)) * ixy1map, axis=2)
            
            
            depth[mask] = 1/inverdepth[mask]


    depth[depth <= 0.02] = pwinverdepth[depth <= 0.02]
    depth[depth == 1e5] = pwinverdepth[depth == 1e5]
    return depth


def test_custom(model, criterion, dataloader, device,  save =False, outputdir = None, vis = False,threshold=(0.5, 0.3, 0.3, 0.3)):
    model.eval()
    pointmaps = []
    segmentations = []
    results = {}
    for iters, (inputs,filename) in enumerate(dataloader):
        image_id = str(iters)
        # set device
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        # forward
        with torch.no_grad():
            x = model(inputs['img'])
        loss, loss_stats = criterion(x)
        
        # post process on output feature map size, and extract planes, lines, plane params instance and plane params pixelwise
        dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise = post_process(x, Mnms=1)

        # reconstruction according to detection results
 
        i = 0
        (_ups, _downs, _attribution, _params_layout), _, ( pfloor, pceiling), walls, dtls = Reconstruction(
            dt_planes[i],
            dt_params3d_instance[i],
            dt_lines[i],
            K=inputs['intri'][i].cpu().numpy(),
            size=(720, 1280),
            threshold=threshold)

        K=inputs['intri'][i].cpu().numpy()
        K[1,1] = -K[1,1]
        new_K = get_scaled_intrinsic(K, (1280,720), (640,360))
        if len(pfloor)>0: 
            pfloor = pfloor[0]
            pfloor[1] = -pfloor[1]
        if len(pceiling)>0:
            pceiling = pceiling[0]
            pceiling[1] = -pceiling[1]

        for j,param in enumerate(_params_layout):
            if param is not None:
                param = param.reshape(4,)
                param[1] = -param[1]
                _params_layout[j] = param

        segmentation = get_segmentation(_ups, _downs, pfloor, pceiling, _params_layout)
        segmentations.append(segmentation)
        pwalls = []
        relations = []
        for param in _params_layout:
            if param is not None:
                pwalls.append(param)
                relations.append(True)
            else:
                relations[-1] = False
        assert len(pwalls) == len(relations)
        depth = get_depth(segmentation, pfloor, pceiling, pwalls, new_K)

        walls[:, :4] = walls[:, :4] * downsample
        walls[:, :4] = np.clip(walls[:, :4], 0, [1279, 719, 1279, 719])
        dtls = np.array(dtls)
        if len(dtls)>0:
            dtls[:, :2] = dtls[:, :2] * downsample
            assert len(dtls) == len(pwalls)-1

        # Convert depth map to 3D point cloud

       
        depth = cv2.resize(depth, (640*2,360*2))
        height, width = depth.shape

        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        z = depth
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / -K[1, 1]
        pointmap = np.stack((x, y, z), axis=-1)
        pointmaps.append(pointmap)
        pwalls = np.array(pwalls)
        
        results[image_id] = {
            "pwalls": pwalls.tolist(),
            "pfloor": pfloor.tolist(),
            "pceiling": pceiling.tolist(),
            "walls": walls.tolist(),
            "relations": relations,
            "line": dtls.tolist(),
        }



    if vis:

        for i in range(len(pointmaps)):
            plt.imshow(segmentations[i])
            plt.figure()
            plt.imshow(pointmaps[i][:,:,2])
            
            dtls = results[str(i)]["line"]
            for j in range(len(dtls)):
                m,b = dtls[j][:2]
                plt.plot([b, m*720+b], [0, 720])
            plt.show()
            
    if save:
        with open(outputdir+"/non_cuboid_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        # np.savez(outputdir+"/pointmaps.npz",pointmaps=pointmaps)
    return results, pointmaps
       

       
def extract_layout(image_list, model, cfg, vis = False, save = False, outputdir = None,threshold=(0.5, 0.3, 0.3, 0.3)):
    dataset = Dust3rDataset(cfg.Dataset.CUSTOM, 'test',files=image_list)

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=cfg.num_workers)

    # create network
    
    # compute loss
    criterion = Loss(cfg.Weights)
    
        
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    criterion.to(device)


    return test_custom(model, criterion, dataloader, device, vis = vis, save = save, outputdir=outputdir,threshold=threshold)

if __name__ == "__main__":

    noncuboid_model = Detector()
    state_dict = torch.load("Structured3D_pretrained.pt",
                                    map_location=torch.device('cpu'))
    noncuboid_model.load_state_dict(state_dict)
    cfg_path = "cfg.yaml"
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
        cfg = EasyDict(config)
    image_list = ["4_rgb_rawlight.png"]
    extract_layout(image_list, noncuboid_model, cfg)

