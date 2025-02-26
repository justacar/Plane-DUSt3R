import os
import sys

HERE_PATH = os.path.normpath(os.path.dirname(__file__))
NONCUBOID_REPO_PATH = os.path.normpath(os.path.join(HERE_PATH, 'noncuboid'))
if os.path.isdir(NONCUBOID_REPO_PATH):
    # workaround for sibling import
    sys.path.insert(0, HERE_PATH)

import numpy as np
import torch
import yaml
from easydict import EasyDict
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import json
from noncuboid.datasets import Dust3rDataset
from noncuboid.models import (Loss, post_process, FilterLine)
from noncuboid.models.reconstruction import MergeNeighborWithSameParam, CommonRegion
# plt.switch_backend('agg')

def ConvertLineFormat(line):
    x1, y1, x2, y2 = line
    m =  (x2 - x1)/(y2 - y1) 
    b = x1 - m * y1
    return m, b



def PreProcess(planes, params, lines, threshold=(0.5, 0.1, 0.1, 0.5)):
    '''
    only process one img per time
    :param planes: a list [[x y x y score cls], ...]
    :param params: a list [[*n d], ...]
    :param lines: a list [[m b score], ...]
    :param threshold: the threshold for wall floor ceiling line
    :return:
    '''
    planes = np.array(planes)
    params = np.array(params)
    lines = np.array(lines)
    
    # select valid detection after nms output
    params = params[planes[:, -1] == 1]
    planes = planes[planes[:, -1] == 1, :-1]
    lines = lines[lines[:, -1] == 1, :-1]
    # split category wall floor ceiling
    walls = planes[planes[:, 5] == 0]
    floor = planes[planes[:, 5] == 1]
    ceiling = planes[planes[:, 5] == 2]
    # split plane params into wall/floor/ceiling param
    pwalls = params[planes[:, 5] == 0]
    pfloor = params[planes[:, 5] == 1]
    pceiling = params[planes[:, 5] == 2]
    # select highest output, at least one plane should be in an image
    hparam = params[planes[:, 4] == np.max(planes[:, 4])][0]
    hplane = planes[planes[:, 4] == np.max(planes[:, 4])][0]
    # select higher score output than threshold
    pwalls = pwalls[walls[:, 4] > threshold[0]]
    pfloor = pfloor[floor[:, 4] > threshold[1]]
    pceiling = pceiling[ceiling[:, 4] > threshold[2]]
    walls = walls[walls[:, 4] > threshold[0]]
    floor = floor[floor[:, 4] > threshold[1]]
    ceiling = ceiling[ceiling[:, 4] > threshold[2]]
    lines = lines[lines[:, 2] > threshold[3]]
    # supposed only one floor and ceiling and floor and ceiling cann't intersection
    if len(floor) > 1:  # at most one floor
        pfloor = pfloor[floor[:, 4] == np.max(floor[:, 4])]
    if len(ceiling) > 1:  # at most one ceiling
        pceiling = pceiling[ceiling[:, 4] == np.max(ceiling[:, 4])]
    if len(pfloor) + len(pceiling) == 2 and len(pwalls) == 0:  # if there are both floor and ceiling, and no walls exist. we select higher score plane for simplify. 
        pfloor = [] if np.max(floor[:, 4]) < np.max(
            ceiling[:, 4]) else pfloor
        pceiling = [] if np.max(floor[:, 4]) >= np.max(
            ceiling[:, 4]) else pceiling
    if len(pfloor) + len(pceiling) + len(pwalls) == 0:  # at least one plane
        if hplane[5] == 0:
            pwalls = np.array([hparam])
            walls = np.array([hplane])
        elif hplane[5] == 1:
            pfloor = np.array([hparam])
        else:
            pceiling = np.array([hparam])

    return walls, pwalls, pfloor, pceiling, lines,floor, ceiling

def get_dtls(common, lines, size, downsample):
    dtls = []
    num = len(common) + 1
    h = size[0] / downsample
    if num > 1:
        for i in range(num-1):
            c = common[i]
            bound = np.array(
                [[c[0], 0], [c[1], 0], [c[0], h], [c[1], h]])  # 4*2
           
            # whether common region exists detection lines
            bound = np.reshape(bound, (4, 1, 2))
            offset = bound[:, :, 0] - lines[:, 0] * \
                bound[:, :, 1] - lines[:, 1]  # 4 * N
            maxv = np.max(offset, axis=0)
            minv = np.min(offset, axis=0)
            inlines = lines[maxv * minv < 0]

            if len(inlines) > 0:
                    # an occulision line exists
                dtl = inlines[inlines[:, 2] == np.max(inlines[:, 2])][0]
            else:
                # others case fail detection
                if c[0] >= 1e5:
                    dtl = np.array([0, c[2], c[3]])
                else:
                    dtl = np.array([0, (c[2]+c[3])/2, 1])
            dtls.append(dtl)
    return np.array(dtls)

def test_custom(model, criterion, dataloader, device,vis = False, save = False, outputfile = None, threshold=(0.5, 0.3, 0.3, 0.3)):
    model.eval()
    import json
    results = {}
    for iters, (inputs, filename) in enumerate(dataloader):
  
        image_id = str(iters)
        # set device
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        # forward
        with torch.no_grad():
            x = model(inputs['img'])
        loss, loss_stats = criterion(x)  # post process on output feature map size, and extract planes, lines, plane params instance and plane params pixelwise
        dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise = post_process(x, Mnms=1)

        size = (720, 1280)
        downsample = 8

        
        # reconstruction according to detection results
        walls, pwalls, pfloor, pceiling, lines, floor, ceiling = PreProcess(dt_planes[0], dt_params3d_instance[0], dt_lines[0], threshold=threshold)
        # merge two neighborhood walls with similar plane params
        walls, pwalls = MergeNeighborWithSameParam(walls, pwalls)

        # define potential intersection region between two neighbor walls
        commons, pwalls = CommonRegion(walls, pwalls, width=size[1] / downsample)

        lines = FilterLine(commons, lines, height=size[0]/downsample)
       
        dtls = get_dtls(commons, lines, size, downsample)

        xyxy = walls[:, :4]
        centerx = np.mean(xyxy[:, [0, 2]], axis=1)
        index = np.argsort(centerx)
        plane = walls[index]

        commons = commons * downsample
        plane[:, :4] = plane[:, :4] * downsample
        plane[:, :4] = np.clip(plane[:, :4], 0, [1279, 719, 1279, 719])
        if len(floor)>0:
            floor[:, :4] = floor[:, :4] * downsample
            floor[:, :4] = np.clip(floor[:, :4], 0, [1279, 719, 1279, 719])
        if len(ceiling)>0:
            ceiling[:, :4] = ceiling[:, :4] * downsample
            ceiling[:, :4] = np.clip(ceiling[:, :4], 0, [1279, 719, 1279, 719])
        if len(dtls)>0:
            dtls[:, :2] = dtls[:, :2] * downsample
        

        to_delete = []
        for i in range(len(plane)):
            x1, y1, x2, y2 = plane[i, :4]
            if x1 >= x2 or y1 >= y2:
                to_delete.append(i)
        
        # Adjust the indices for dtls and commons which are based on i and i+1 elements of plane
        to_delete_dtls_commons = [i for i in to_delete if i < len(dtls)]
        
        # Delete the invalid entries from plane, dtls, and commons
        plane = np.delete(plane, to_delete, axis=0)
        dtls = np.delete(dtls, to_delete_dtls_commons, axis=0)
        commons = np.delete(commons, to_delete_dtls_commons, axis=0)
                


        # Store results in a dictionary
        results[image_id] = {
            "commons": commons.tolist(),
            "floor": floor.tolist(),
            "ceiling": ceiling.tolist(),
            "plane": plane.tolist(),
            "line": dtls.tolist()
        }
        if vis:
            img = Image.open(filename[0])
            plt.imshow(img.resize((1280,720)))
            

            for i in range(len(plane)):
                xyxy = plane[:,:4]
                plt.plot([plane[i, 0], plane[i, 2]], [plane[i, 1], plane[i, 1]],linewidth=3)
                plt.plot([plane[i, 0], plane[i, 0]], [plane[i, 1], plane[i, 3]],linewidth=3)
                plt.plot([plane[i, 0], plane[i, 2]], [plane[i, 3], plane[i, 3]],linewidth=3)
                plt.plot([plane[i, 2], plane[i, 2]], [plane[i, 1], plane[i, 3]],linewidth=3)
            if len(floor)>0:
                plt.plot([floor[0, 0], floor[0, 2]], [floor[0, 1], floor[0, 1]],linewidth=3)
                plt.plot([floor[0, 0], floor[0, 0]], [floor[0, 1], floor[0, 3]],linewidth=3)
                plt.plot([floor[0, 0], floor[0, 2]], [floor[0, 3], floor[0, 3]],linewidth=3)
                plt.plot([floor[0, 2], floor[0, 2]], [floor[0, 1], floor[0, 3]],linewidth=3)
            if len(ceiling)>0:
                plt.plot([ceiling[0, 0], ceiling[0, 2]], [ceiling[0, 1], ceiling[0, 1]],linewidth=3)
                plt.plot([ceiling[0, 0], ceiling[0, 0]], [ceiling[0, 1], ceiling[0, 3]],linewidth=3)
                plt.plot([ceiling[0, 0], ceiling[0, 2]], [ceiling[0, 3], ceiling[0, 3]],linewidth=3)
                plt.plot([ceiling[0, 2], ceiling[0, 2]], [ceiling[0, 1], ceiling[0, 3]],linewidth=3)
            # for common in commons:
            #     plt.plot([common[0], common[0]], [0, 720])
            #     plt.plot([common[1], common[1]], [0, 720])
            # for i in range(len(dtls)):
            #     m,b = dtls[i][:2]
            #     plt.plot([b, m*720+b], [0, 720])
            plt.show()
           
            # plt.axis("off")
          
            plt.close()


    # Write results to a JSON file
    if save:
        with open(outputfile, 'w') as f:
            json.dump(results, f, indent=4)
    return results

def extract_plane(image_list, model, cfg, vis = False, save = False, filename = None, threshold=(0.5, 0.3, 0.3, 0.3)):
    dataset = Dust3rDataset(cfg.Dataset.CUSTOM, 'test',files=image_list)

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=cfg.num_workers)
    
    # compute loss
    criterion = Loss(cfg.Weights)
      
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    criterion.to(device)

    return test_custom(model, criterion, dataloader, device, vis, save = save, outputfile = filename, threshold=threshold)
   
if __name__ == '__main__':
    import glob
    path = ""
    image_list = sorted(glob.glob(os.path.join(path, '*_rgb_rawlight.png')))    
    extract_plane(image_list, "Structured3D_pretrained.pt")

