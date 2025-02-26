import numpy as np
import json
import glob
import os
import copy
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import  Polygon
from scipy.optimize import linear_sum_assignment,fsolve
from utils import parse_camera_info

def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def _validate_colormap(colormap, n_labels):
    if colormap is None:
        colormap = label_colormap(n_labels)
    else:
        assert colormap.shape == (colormap.shape[0], 3), \
            'colormap must be sequence of RGB values'
        assert 0 <= colormap.min() and colormap.max() <= 1, \
            'colormap must ranges 0 to 1'
    return colormap

def match_by_Hungarian(gt, pred):
    n = len(gt)
    m = len(pred)
    gt = np.array(gt)
    pred = np.array(pred)
    valid = (gt.sum(0) > 0).sum()
    if m == 0:
        raise IOError
    else:
        gt = gt[:, np.newaxis, :, :]
        pred = pred[np.newaxis, :, :, :]
        cost = np.sum((gt+pred) == 2, axis=(2, 3))  # n*m
        row, col = linear_sum_assignment(-1 * cost)
        inter = cost[row, col].sum()
        PE = inter / valid
        return 1 - PE


def evaluate(gtseg, gtdepth, preseg, predepth, evaluate_2D=True, evaluate_3D=True):
    image_iou, image_pe, merror_edge, rmse, us_rmse = 0, 0, 0, 0, 0
    if evaluate_2D:
        # Parse GT polys
        gt_polys_masks = []
        h, w = gtseg.shape
        gt_polys_edges_mask = np.zeros((h, w))
        edge_thickness = 1
        gt_valid_seg = np.ones((h, w))
        labels = np.unique(gtseg)
        for i, label in enumerate(labels):
            gt_poly_mask = gtseg == label
            if label == -1:
                gt_valid_seg[gt_poly_mask] = 0  # zero pad region
            else:
                contours_, hierarchy = cv2.findContours(gt_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(gt_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                gt_polys_masks.append(gt_poly_mask.astype(np.int32))

        def sortPolyBySize(mask):
            return mask.sum()
        gt_polys_masks.sort(key=sortPolyBySize, reverse=True)

        # Parse predictions
        pred_polys_masks = []
        pred_polys_edges_mask = np.zeros((h, w))
        pre_invalid_seg = np.zeros((h, w))
        labels = np.unique(preseg)
        for i, label in enumerate(labels):
            pred_poly_mask = np.logical_and(preseg == label, gt_valid_seg == 1)
            if pred_poly_mask.sum() == 0:
                continue
            if label == -1:
                # zero pad and infinity region
                pre_invalid_seg[pred_poly_mask] = 1
            else:
                contours_, hierarchy = cv2.findContours(pred_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_SIMPLE
                cv2.polylines(pred_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                pred_polys_masks.append(pred_poly_mask.astype(np.int32))
        if len(pred_polys_masks) == 0.:
            pred_polys_edges_mask[edge_thickness:-
                                  edge_thickness, edge_thickness:-edge_thickness] = 1
            pred_polys_edges_mask = 1 - pred_polys_edges_mask
            pred_poly_mask = np.ones((h, w))
            pred_polys_masks = [pred_poly_mask]

        pred_polys_masks_cand = copy.copy(pred_polys_masks)
        # Assign predictions to ground truth polygons
        ordered_preds = []
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            best_iou_score = 0.3
            best_pred_ind = None
            best_pred_poly_mask = None
            if len(pred_polys_masks_cand) == 0:
                break
            for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand):
                gt_pred_add = gt_poly_mask + pred_poly_mask
                inter = np.equal(gt_pred_add, 2.).sum()
                union = np.greater(gt_pred_add, 0.).sum()
                iou_score = inter / union

                if iou_score > best_iou_score:
                    best_iou_score = iou_score
                    best_pred_ind = pred_ind
                    best_pred_poly_mask = pred_poly_mask
            ordered_preds.append(best_pred_poly_mask)

            pred_polys_masks_cand = [pred_poly_mask for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand)
                                     if pred_ind != best_pred_ind]
            if best_pred_poly_mask is None:
                continue

        ordered_preds += pred_polys_masks_cand
        class_num = max(len(ordered_preds), len(gt_polys_masks))
        colormap = _validate_colormap(None, class_num + 1)

        # Generate GT poly mask
        gt_layout_mask = np.zeros((h, w))
        gt_layout_mask_colored = np.zeros((h, w, 3))
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            gt_layout_mask = np.maximum(
                gt_layout_mask, gt_poly_mask * (gt_ind + 1))
            gt_layout_mask_colored += gt_poly_mask[:,
                                                   :, None] * colormap[gt_ind + 1]

        # Generate pred poly mask
        pred_layout_mask = np.zeros((h, w))
        pred_layout_mask_colored = np.zeros((h, w, 3))
        for pred_ind, pred_poly_mask in enumerate(ordered_preds):
            if pred_poly_mask is not None:
                pred_layout_mask = np.maximum(
                    pred_layout_mask, pred_poly_mask * (pred_ind + 1))
                pred_layout_mask_colored += pred_poly_mask[:,
                                                           :, None] * colormap[pred_ind + 1]

        # Calc IOU
        ious = []
        for layout_comp_ind in range(1, len(gt_polys_masks) + 1):
            inter = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                   np.equal(pred_layout_mask, layout_comp_ind)).sum()
            fp = np.logical_and(np.not_equal(gt_layout_mask, layout_comp_ind),
                                np.equal(pred_layout_mask, layout_comp_ind)).sum()
            fn = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                np.not_equal(pred_layout_mask, layout_comp_ind)).sum()
            union = inter + fp + fn
            iou = inter / union
            ious.append(iou)

        image_iou = sum(ious) / class_num

        # Calc PE
        image_pe = 1 - np.equal(gt_layout_mask[gt_valid_seg == 1],
                                pred_layout_mask[gt_valid_seg == 1]).sum() / (np.sum(gt_valid_seg == 1))
        # Calc PE by Hungarian
        image_pe_hung = match_by_Hungarian(gt_polys_masks, pred_polys_masks)
        # Calc edge error
        # ignore edges at image borders
        img_bound_mask = np.zeros_like(pred_polys_edges_mask)
        img_bound_mask[10:-10, 10:-10] = 1

        pred_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - pred_polys_edges_mask)).astype(np.uint8),
                                                cv2.DIST_L2, 3)
        gt_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - gt_polys_edges_mask)).astype(np.uint8),
                                              cv2.DIST_L2, 3)

        chamfer_dist = pred_polys_edges_mask * gt_dist_trans + \
            gt_polys_edges_mask * pred_dist_trans
        if np.sum(
            np.greater(img_bound_mask * (gt_polys_edges_mask), 0))>0:
            merror_edge = 0.5 * np.sum(chamfer_dist) / np.sum(
                np.greater(img_bound_mask * (gt_polys_edges_mask), 0)) 
        else:
            merror_edge = 0

    # Evaluate in 3D
    if evaluate_3D:
        max_depth = 50
        gt_layout_depth_img_mask = np.greater(gtdepth, 0.)
        gt_layout_depth_img = 1. / gtdepth[gt_layout_depth_img_mask]
        gt_layout_depth_img = np.clip(gt_layout_depth_img, 0, max_depth)
        gt_layout_depth_med = np.median(gt_layout_depth_img)
        # max_depth = np.max(gt_layout_depth_img)
        # may be max_depth should be max depth of all scene
        predepth[predepth == 0] = 1 / max_depth
        pred_layout_depth_img = 1. / predepth[gt_layout_depth_img_mask]
        pred_layout_depth_img = np.clip(pred_layout_depth_img, 0, max_depth)
        pred_layout_depth_med = np.median(pred_layout_depth_img)

        # Calc MSE
        ms_error_image = (pred_layout_depth_img - gt_layout_depth_img) ** 2
        rmse = np.sqrt(np.sum(ms_error_image) /
                       np.sum(gt_layout_depth_img_mask))

        # Calc up to   MSE
        if np.isnan(pred_layout_depth_med) or pred_layout_depth_med == 0:
            d_scale = 1.
        else:
            d_scale = gt_layout_depth_med / pred_layout_depth_med
        us_ms_error_image = (
            d_scale * pred_layout_depth_img - gt_layout_depth_img) ** 2
        us_rmse = np.sqrt(np.sum(us_ms_error_image) /
                          np.sum(gt_layout_depth_img_mask))

    return image_iou, image_pe, merror_edge, rmse, us_rmse#, image_pe_hung


def CalculateFakePlane(line, K_inv):
    def func(variable, ray):
        x, y, z = variable
        a = ray[0, 0] * x + ray[1, 0] * y + ray[2, 0] * z
        b = ray[0, 1] * x + ray[1, 1] * y + ray[2, 1] * z
        c = x * x + y * y + z * z - 1
        return [a, b, c]
    ones = np.ones([2, 1])
    point = np.concatenate([line, ones], axis=1).T
    ray = np.dot(K_inv, point)  # 3*2
    result = fsolve(func, np.array([0, 0, 0]), args=(ray))
    result = [*result, 0]
    return result

def CalculateIntersectionPoint(p0, p1, p2, K, UD=0, downsample=1, size=(720, 1280)):
    # nx+d=0 plane params is for original resolution
    if len(p2) == 0:  # not exist floor or ceiling
        K_inv = np.linalg.inv(K)
        if UD == 0:  # floor
            fake_line = np.array([[0, size[0]-1], [size[1]-1, size[0]-1]])
            p2 = CalculateFakePlane(fake_line, K_inv)
        else:  # ceiling
            fake_line = np.array([[0, 0], [size[1]-1, 0]])
            p2 = CalculateFakePlane(fake_line, K_inv)
    assert len(p0)!=0 and len(p1)!=0 and len(p2)!=0
    assert np.isnan(p0).any() == False
    assert np.isnan(p1).any() == False
    assert np.isnan(p2).any() == False
    coefficient = np.array([p0, p1, p2])
    A = coefficient[:, :3]
    B = -1 * coefficient[:, 3]
    res = np.linalg.solve(A, B)
    # project 3d to 2d
    point_3d = res.reshape(3, 1)
    point_2d = np.dot(K, point_3d) / point_3d[2, 0]
    point_2d = point_2d[:2, 0] / downsample
    return point_2d

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

def reconstruction(pwalls,relationship, dtls, pceiling, pfloor,K,downsample=2, size=(720, 1280)):
    ups = []
    downs = []
    pwalls_output = []
    K_inv = np.linalg.inv(K)
    for i in range(len(pwalls)-1):
        p0 = pwalls[i]
        p1 = pwalls[i+1]
        if relationship[i] == 0:
            

            dtl = dtls[i]
            fake_line = np.array([[dtl[1], 0], [dtl[0]+dtl[1], 1]]) 
            fake_plane = CalculateFakePlane(fake_line, K_inv)
            point0 = CalculateIntersectionPoint(
                p0, fake_plane, pfloor, K, 0,downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                p0, fake_plane, pceiling, K, 1,downsample=downsample, size=size)
            point2 = CalculateIntersectionPoint(
                fake_plane, p1, pfloor, K, 0,downsample=downsample, size=size)
            point3 = CalculateIntersectionPoint(
                fake_plane, p1, pceiling, K, 1,downsample=downsample, size=size)
            downs.append(point0)
            downs.append(point2)
            ups.append(point1)
            ups.append(point3)
            pwalls_output.append(p0)
            pwalls_output.append(None)
        else: # two walls intersect

            point0 = CalculateIntersectionPoint(
                p0, p1, pfloor, K, 0,downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                p0, p1, pceiling, K, 1,downsample=downsample, size=size)
            downs.append(point0)
            ups.append(point1)
            pwalls_output.append(p0)

    if len(pwalls) > 0:  # determine the left and right boundary with image boundary.
        if len(pwalls) == 1:
            # left boundary
            fake_line = np.array([[0, 0], [0, size[0]]])
            fake_plane = CalculateFakePlane(fake_line, K_inv)
            point0 = CalculateIntersectionPoint(
                fake_plane, pwalls[0], pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                fake_plane, pwalls[0], pceiling, K, 1, downsample=downsample, size=size)
           
            downs = [point0, *downs]
            ups = [point1, *ups]
            # right boundary
            fake_line = np.array([[size[1], 0], [size[1], size[0]]])
            fake_plane = CalculateFakePlane(fake_line, K_inv)
            point0 = CalculateIntersectionPoint(
                fake_plane, pwalls[-1], pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                fake_plane, pwalls[-1], pceiling, K, 1, downsample=downsample, size=size)
            downs.append(point0)
            ups.append(point1)
            pwalls_output.append(pwalls[0])
           
        else:
            # left boundary
            fake_line = np.array([ups[0], downs[0]])
            m = (fake_line[0, 0] - fake_line[1, 0]) / \
                (fake_line[0, 1] - fake_line[1, 1])
            left_line = np.zeros_like(fake_line)

            if m < 0:
                left_line[0] = [0, 0]
            else:
                left_line[0] = [0, size[0]]
            b = left_line[0, 0] - m * left_line[0, 1]
            left_line[1] = [m*size[0]/2 + b, size[0]/2]

            fake_plane = CalculateFakePlane(left_line, K_inv)

            point0 = CalculateIntersectionPoint(
                fake_plane, pwalls[0], pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                fake_plane, pwalls[0], pceiling, K, 1, downsample=downsample, size=size)
            downs = [point0, *downs]
            ups = [point1, *ups]
            # right boundary
            fake_line = np.array([ups[-1], downs[-1]])
            m = (fake_line[0, 0] - fake_line[1, 0]) / \
                (fake_line[0, 1] - fake_line[1, 1])
            right_line = np.zeros_like(fake_line)

            if m < 0:
                right_line[0] = [size[1], size[0]]
            else:
                right_line[0] = [size[1], 0]
            b = right_line[0, 0] - m * right_line[0, 1]
            right_line[1] = [m*size[0]/2 + b, size[0]/2]

            fake_plane = CalculateFakePlane(right_line, K_inv)
            point0 = CalculateIntersectionPoint(
                fake_plane, pwalls[-1], pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                fake_plane, pwalls[-1], pceiling, K, 1, downsample=downsample, size=size)
            downs.append(point0)
            ups.append(point1)
            pwalls_output.append(pwalls[-1])
    else:
        assert len(pfloor) + len(pceiling) == 1
    return ups,downs,pwalls_output


def dataload(layout_name, ratio_h, inh, inw):
    with open(layout_name, 'r') as f:
        anno_layout = json.load(f)
        junctions = anno_layout['junctions']
        planes = anno_layout['planes']

        coordinates = []
        for k in junctions:
            coordinates.append(k['coordinate'])
        coordinates = np.array(coordinates) / ratio_h

        pparams = []
        labels = []
        segs = -1 * np.ones([inh, inw])
     
        i = 0
        for pp in planes:
            if len(pp['visible_mask']) != 0:
                if pp['type'] == 'wall':
                    cout = coordinates[pp['visible_mask'][0]]
                    polygon = Polygon(cout)
                    if polygon.area >= 1000:
                        cout = cout.astype(np.int32)
                        cv2.fillPoly(segs, [cout], color=i)
                        pparams.append([*pp['normal'], pp['offset'] / 1000.])
                        labels.append(0)
                        i += 1
                else:
                    for v in pp['visible_mask']:
                        cout = coordinates[v]
                        polygon = Polygon(cout)
                        if polygon.area > 1000:
                            cout = cout.astype(np.int32)
                            cv2.fillPoly(segs, [cout], color=i)
                            pparams.append([*pp['normal'], pp['offset'] / 1000.])
                            if pp['type'] == 'floor':
                                labels.append(1)
                            else:
                                labels.append(2)
                            i += 1
   
    return pparams, segs


def is_matched(plane_param1, plane_param2, angle_threshold_deg=10, offset_threshold=0.15):
    normal1, offset1 = plane_param1[:3], plane_param1[3]
    normal2, offset2 = plane_param2[:3], plane_param2[3]
    
    # Calculate the angle between the two normals
    cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    angle_deg = min(angle_deg, 180-angle_deg)
    # Calculate the difference in offsets
    offset_diff = np.abs(offset1 - offset2)
    # print(angle_deg, offset_diff)
    # Check if the planes match based on the angle and offset difference
    if angle_deg < angle_threshold_deg and offset_diff < offset_threshold:
        return True
    else:
        return False


def extract_scene_planes(room_path, RT0):
    plane_dict = {}

    for position_ids in os.listdir(room_path):
        layout_path = os.path.join(room_path, position_ids, 'layout.json')
        camera_path = os.path.join(room_path, position_ids, 'camera_pose.txt')
        with open(layout_path, 'r') as f:
            layout_info = json.load(f)
        RT, _ = parse_camera_info(np.loadtxt(camera_path), 720, 1280)
        
        with open(os.path.join(room_path, position_ids, 'layout.json'), 'r') as f:
            layout_info = json.load(f)
            for plane in layout_info["planes"]:
                plane_id = plane["ID"]
                if plane_id in plane_dict:
                    continue
                pparam = np.concatenate([plane["normal"], [plane["offset"]]])
                pparam[-1] = pparam[-1] /1000
                transform_matrix = np.linalg.inv(RT0) @ RT
                pparam_trans = pparam @ np.linalg.inv(transform_matrix)

                plane_dict[plane_id] = pparam_trans
    return plane_dict

def extract_scene_planes_view(room_path, RT0, view_count):
    plane_dict = {}

    for position_ids in os.listdir(room_path):
        if int(position_ids) >= view_count:
            continue
        layout_path = os.path.join(room_path, position_ids, 'layout.json')
        camera_path = os.path.join(room_path, position_ids, 'camera_pose.txt')
        with open(layout_path, 'r') as f:
            layout_info = json.load(f)
        RT, _ = parse_camera_info(np.loadtxt(camera_path), 720, 1280)
        
        with open(os.path.join(room_path, position_ids, 'layout.json'), 'r') as f:
            layout_info = json.load(f)
            for plane in layout_info["planes"]:
                plane_id = plane["ID"]
                if plane_id in plane_dict:
                    continue
                pparam = np.concatenate([plane["normal"], [plane["offset"]]])
                pparam[-1] = pparam[-1] /1000
                transform_matrix = np.linalg.inv(RT0) @ RT
                pparam_trans = pparam @ np.linalg.inv(transform_matrix)

                plane_dict[plane_id] = pparam_trans
    return plane_dict
        
        

def metric_geodust3r(plane_detection, dust3r_output, node_info, image_list, mode = "",vis = False, save = False, filedir = None, metric = False):
    floor = node_info['floor_pparam']
    ceiling = node_info['ceiling_pparam']

    h,w = 360,640
    K= np.array([[762, 0, 640], [0, 762, 360], [0, 0, 1]],
                dtype=np.float32)
    new_K = get_scaled_intrinsic(K, (1280,720), (640,360))


    plane_poses = dust3r_output["poses"]

    identity_matrix = np.eye(3)

    closest_pose_index = 0
    min_rotation_diff = float('inf')
    min_translation_diff = float('inf')
    for i, pose in enumerate(plane_poses):
        rotation_matrix = pose[:3, :3]
        translation_vector = pose[:3, 3]

        rotation_diff = np.linalg.norm(rotation_matrix - identity_matrix)
        translation_diff = np.linalg.norm(translation_vector)

        if rotation_diff < min_rotation_diff or (rotation_diff == min_rotation_diff and translation_diff < min_translation_diff):
            min_rotation_diff = rotation_diff
            min_translation_diff = translation_diff
            closest_pose_index = i
    room_path = "/".join(image_list[0].split('/')[:-2])
    RT0, _ = parse_camera_info(np.loadtxt(f"{room_path}/{closest_pose_index}/camera_pose.txt"), 720, 1280)
    if metric:
        average_scale  = 1
    else:
        scales = []
        pose0 = plane_poses[closest_pose_index]
        
        #TODO do not include the coordinate image
        for i in range(len(plane_poses)):
            
            RT_pred = np.linalg.inv(plane_poses[i]) @ pose0
            camera_info = np.loadtxt(f"{room_path}/{i}/camera_pose.txt")
            RT, _ = parse_camera_info(camera_info, 720, 1280)
            RT_gt = np.linalg.inv(RT) @ RT0
            # Extract translation vectors
            t_pred = RT_pred[:3, 3]
            t_gt = RT_gt[:3, 3]
            # print(RT_pred)
            # print(RT_gt)
            if i == closest_pose_index or sum(t_pred) == 0:
                continue
            # Compute scale as the ratio of norms of translation vectors
            scale = np.linalg.norm(t_gt) / np.linalg.norm(t_pred)
            scales.append(scale)

        #if no scale, give a default value
        if len(scales) == 0:
            average_scale = 11
        else:
            average_scale = np.mean(scales)


    wall_relationship = node_info["wall_relationship"]


    metric_results = []
    depths = []
    segemantations = []

    for image_path in image_list:
        img_id = image_path.split('/')[-2]
        depth_ori = cv2.imread(image_path.replace("rgb_rawlight.png","plane_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        pparams, segs = dataload(image_path.replace("rgb_rawlight.png","layout.json"), 720/h, h, w)
        
        pts = dust3r_output['pts3d'][int(img_id)]
        pts = pts.reshape(-1,3)
        if img_id != closest_pose_index:
            transform_matrix = np.linalg.inv(dust3r_output['poses'][int(img_id)]) @ dust3r_output['poses'][closest_pose_index]
            
        else:
            transform_matrix = np.eye(4)
    
        segmentation = -1 * np.ones([h, w])  # [0: ceiling, 1: floor, 2...:walls]

        if len(floor)>0:
            cur_floor = floor @ np.linalg.inv(transform_matrix) 
            cur_floor[3] = cur_floor[3] * average_scale
        else:
            cur_floor = []
        if len(ceiling)>0:
            cur_ceiling = ceiling @ np.linalg.inv(transform_matrix) 
            cur_ceiling[3] = cur_ceiling[3] * average_scale
        else:
            cur_ceiling = []

        pwalls = []
        plane_indexs = node_info["planes"][img_id] #TODO
        for index in plane_indexs:
            plane_pparam = node_info["global_plane_info"][index]["pparam"]
            assert index == node_info["global_plane_info"][index]["index"]
            
            pwall = plane_pparam @ np.linalg.inv(transform_matrix)
            pwall[3] = pwall[3] * average_scale
            pwalls.append(pwall)
            
    
        ups,downs,pwalls_output = reconstruction(pwalls,wall_relationship[int(img_id)],plane_detection[img_id]["line"],cur_ceiling,cur_floor, K)
        ups = np.array(ups).astype(np.int32)
        downs = np.array(downs).astype(np.int32)

        
        if len(ceiling)>0: 
            minuy = min(np.min(ups[:, 1]) - 10, -1)
            cv2.fillPoly(img=segmentation, pts=np.array(
            [[[ups[0, 0], minuy], *ups, [ups[-1, 0], minuy]]]), color=0)
        if len(floor)>0: 
            maxdy = max(np.max(downs[:, 1]) + 10, h+1)
            cv2.fillPoly(img=segmentation, pts=np.array(
                [[[downs[0, 0], maxdy], *downs, [downs[-1, 0], maxdy]]]), color=1)

        assert len(ups) == len(pwalls_output) + 1
        j = -1
        for i in range(len(ups)-1):
            u0 = ups[i]
            u1 = ups[i+1]
            
            if pwalls_output[i] is None:
                continue
            d0 = downs[i]
            d1 = downs[i+1]

            j = j + 1

            cv2.fillPoly(img=segmentation, pts=np.array(
                [[u0, d0, d1, u1]]), color=j+2)


        pwinverdepth = np.ones_like(segmentation) * 1e5
        # pixelwise
        
        depth = np.ones_like(segmentation) * 1e5
        labels = np.unique(segmentation)
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
                param = cur_ceiling
            elif label == 1:
                param = cur_floor
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
        if vis:
            pass
            plt.imshow(depth)
            plt.figure()
            plt.imshow(depth_ori)
            # plt.figure()
            # plt.imshow(segs)
           
            # plt.imshow(segmentation)
            plt.show()
        depths.append(depth)
        segemantations.append(segmentation)
         
    
        image_iou, image_pe, merror_edge, rmse, us_rmse= evaluate(segs,cv2.resize(depth_ori, (640,360)), segmentation,depth, True, True)
        metric_results.append([image_iou, image_pe, merror_edge, rmse, us_rmse])


    # scene metric
    gt_planes = extract_scene_planes(room_path, RT0)
    gt_planes = list(gt_planes.values())
    pred_planes = []
    if len(node_info["floor_pparam"]) > 0:
        floor_pparam = node_info["floor_pparam"]
        floor_pparam[-1] = floor_pparam[-1] * average_scale
        pred_planes.append(floor_pparam)
    if len(node_info["ceiling_pparam"]) > 0:
        ceiling_pparam = node_info["ceiling_pparam"]
        ceiling_pparam[-1] = ceiling_pparam[-1] * average_scale
        pred_planes.append(node_info["ceiling_pparam"])
    for plane_info in node_info["global_plane_info"]:
        plane_pparam = plane_info["pparam"]
        plane_pparam[-1] = plane_pparam[-1] * average_scale
        pred_planes.append(plane_pparam)

    hit = np.zeros(len(gt_planes), np.bool_)
    tp = 0
  

    for i in range(len(pred_planes)):
        for j in range(len(gt_planes)):
            if is_matched(pred_planes[i], gt_planes[j]) and not hit[j]:
                tp += 1
                hit[j] = True
                break
            
    
    
    precision = tp / len(pred_planes)
    recall = tp / len(gt_planes)

    if save:
        with open(os.path.join(filedir, 'metric_results.txt'), 'w') as file:
            for result in metric_results:
                file.write(str(result) + '\n')
            file.write(f"precision: {precision}, recall: {recall}\n")
        # for i,depth in enumerate(depths):
        #     depth_input_uint16 = (depth * 1000).astype(np.uint16)  # Scale depth to millimeters and convert to uint16
        #     depth_filename = os.path.join(filedir, f"depth_pred_{i}.png")
        #     cv2.imwrite(depth_filename, depth_input_uint16)  # Save depth image as PNG
        for i,seg in enumerate(segemantations):
            # cv2.imwrite(os.path.join(filedir, f"seg_pred_{i}.png"), seg)

            plt.imshow(seg)
            plt.savefig(os.path.join(filedir, f"seg_pred_vis_{i}.png"))
            plt.close()
            

    return metric_results, precision, recall

def metric_mast3r(plane_detection, mast3r_output, node_info, image_list, vis = False, save = False, filedir = None):
    room_path = "/".join(image_list[0].split('/')[:-2])
    floor = node_info['floor_pparam']
    ceiling = node_info['ceiling_pparam']

    h,w = 360,640
    K= np.array([[762, 0, 640], [0, 762, 360], [0, 0, 1]],
                dtype=np.float32)
    new_K = get_scaled_intrinsic(K, (1280,720), (640,360))


    poses = mast3r_output
    pose0 = poses[0]
    RT0, _ = parse_camera_info(np.loadtxt(f"{room_path}/{0}/camera_pose.txt"), 720, 1280)


    wall_relationship = node_info["wall_relationship"]


    metric_results = []
    depths = []
    segemantations = []

    for image_path in image_list:
        img_id = image_path.split('/')[-2]
        depth_ori = cv2.imread(image_path.replace("rgb_rawlight.png","plane_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        # depth = cv2.resize(depth,(512,288))
        pparams, segs = dataload(image_path.replace("rgb_rawlight.png","layout.json"), 720/h, h, w)
        
        
        transform_matrix = np.linalg.inv(poses[int(img_id)] ) @ pose0
    
        segmentation = -1 * np.ones([h, w])  # [0: ceiling, 1: floor, 2...:walls]

        if len(floor)>0:
            cur_floor = floor @ np.linalg.inv(transform_matrix) 
        else:
            cur_floor = []
        if len(ceiling)>0:
            cur_ceiling = ceiling @ np.linalg.inv(transform_matrix) 
        else:
            cur_ceiling = []

        pwalls = []
        plane_indexs = node_info["planes"][img_id] #TODO
        for index in plane_indexs:
            plane_pparam = node_info["global_plane_info"][index]["pparam"]
            assert index == node_info["global_plane_info"][index]["index"]
            
            pwall = plane_pparam @ np.linalg.inv(transform_matrix)
            pwalls.append(pwall)
        # print(pwalls) 
        # print(wall_relationship[int(img_id)])
        ups,downs,pwalls_output = reconstruction(pwalls,wall_relationship[int(img_id)],plane_detection[img_id]["line"],cur_ceiling,cur_floor, K)
        ups = np.array(ups).astype(np.int32)
        downs = np.array(downs).astype(np.int32)

        
        
        if len(ceiling)>0: 
            minuy = min(np.min(ups[:, 1]) - 10, -1)
            cv2.fillPoly(img=segmentation, pts=np.array(
            [[[ups[0, 0], minuy], *ups, [ups[-1, 0], minuy]]]), color=0)
        if len(floor)>0: 
            maxdy = max(np.max(downs[:, 1]) + 10, h+1)
            cv2.fillPoly(img=segmentation, pts=np.array(
                [[[downs[0, 0], maxdy], *downs, [downs[-1, 0], maxdy]]]), color=1)

        assert len(ups) == len(pwalls_output) + 1
        j = -1
        for i in range(len(ups)-1):
            u0 = ups[i]
            u1 = ups[i+1]
            
            if pwalls_output[i] is None:
                continue
            d0 = downs[i]
            d1 = downs[i+1]

            j = j + 1

            cv2.fillPoly(img=segmentation, pts=np.array(
                [[u0, d0, d1, u1]]), color=j+2)


        pwinverdepth = np.ones_like(segmentation) * 1e5
        # pixelwise
        
        depth = np.ones_like(segmentation) * 1e5
        labels = np.unique(segmentation)
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
                param = cur_ceiling
            elif label == 1:
                param = cur_floor
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
        if vis:
            pass
            plt.imshow(depth)
            plt.figure()
            plt.imshow(depth_ori)
            plt.figure()
            plt.imshow(segs)
            plt.figure()
            plt.imshow(segmentation)
            plt.show()
        depths.append(depth)
        segemantations.append(segmentation)
         
    
        image_iou, image_pe, merror_edge, rmse, us_rmse= evaluate(segs,cv2.resize(depth_ori, (640,360)), segmentation,depth, True, True)
        metric_results.append([image_iou, image_pe, merror_edge, rmse, us_rmse])


    # scene metric
    gt_planes = extract_scene_planes(room_path, RT0)
    gt_planes = list(gt_planes.values())
    pred_planes = []
    if len(node_info["floor_pparam"]) > 0:
        floor_pparam = node_info["floor_pparam"]
        pred_planes.append(floor_pparam)
    if len(node_info["ceiling_pparam"]) > 0:
        ceiling_pparam = node_info["ceiling_pparam"]
        pred_planes.append(ceiling_pparam)
    for plane_info in node_info["global_plane_info"]:
        plane_pparam = plane_info["pparam"]
        pred_planes.append(plane_pparam)

    hit = np.zeros(len(gt_planes), np.bool_)
    tp = 0
  

    for i in range(len(pred_planes)):
        for j in range(len(gt_planes)):
            if is_matched(pred_planes[i], gt_planes[j]) and not hit[j]:
                tp += 1
                hit[j] = True
                break
            
    
    
    precision = tp / len(pred_planes)
    recall = tp / len(gt_planes)

    if save:
        with open(os.path.join(filedir, 'metric_results.txt'), 'w') as file:
            for result in metric_results:
                file.write(str(result) + '\n')
            file.write(f"precision: {precision}, recall: {recall}\n")
        # for i,depth in enumerate(depths):
        #     depth_input_uint16 = (depth * 1000).astype(np.uint16)  # Scale depth to millimeters and convert to uint16
        #     depth_filename = os.path.join(filedir, f"depth_pred_{i}.png")
        #     cv2.imwrite(depth_filename, depth_input_uint16)  # Save depth image as PNG
        for i,seg in enumerate(segemantations):
            # cv2.imwrite(os.path.join(filedir, f"seg_pred_{i}.png"), seg)

            plt.imshow(seg)
            plt.savefig(os.path.join(filedir, f"seg_pred_vis_{i}.png"))
            plt.close()
            

    return metric_results, precision, recall


def metric_cad(plane_detection, mast3r_output, node_info, image_list, vis = False, save = False, filedir = None):
    room_path = "/".join(image_list[0].split('/')[:-2])
    floor = node_info['floor_pparam']
    ceiling = node_info['ceiling_pparam']

    h,w = 360,640
    K= np.array([[762, 0, 640], [0, 762, 360], [0, 0, 1]],
                dtype=np.float32)
    new_K = get_scaled_intrinsic(K, (1280,720), (640,360))


    poses = mast3r_output
    pose0 = poses[0]
    RT0, _ = parse_camera_info(np.loadtxt(f"{room_path}/{0}/camera_pose.txt"), 720, 1280)


    wall_relationship = node_info["wall_relationship"]


    metric_results = []
    depths = []
    segemantations = []

    for image_path in image_list:
        img_id = image_path.split('/')[-2]
        depth_ori = cv2.imread(image_path.replace("rgb_rawlight.png","plane_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        # depth = cv2.resize(depth,(512,288))
        _, segs = dataload(image_path.replace("rgb_rawlight.png","layout.json"), 720/h, h, w)
        
        
        transform_matrix = np.linalg.inv(poses[int(img_id)] ) @ pose0
    
        segmentation = -1 * np.ones([h, w])  # [0: ceiling, 1: floor, 2...:walls]

        if len(floor)>0:
            cur_floor = floor @ np.linalg.inv(transform_matrix) 
        else:
            cur_floor = []
        if len(ceiling)>0:
            cur_ceiling = ceiling @ np.linalg.inv(transform_matrix) 
        else:
            cur_ceiling = []

        pwalls = []
        plane_indexs = node_info["planes"][img_id] #TODO
        for index in plane_indexs:
            plane_pparam = node_info["global_plane_info"][index]["pparam"]
            assert index == node_info["global_plane_info"][index]["index"]
            
            pwall = plane_pparam @ np.linalg.inv(transform_matrix)
            pwalls.append(pwall)
        # print(pwalls) 
        # print(wall_relationship[int(img_id)])
        ups,downs,pwalls_output = reconstruction(pwalls,wall_relationship[int(img_id)],plane_detection[img_id]["line"],cur_ceiling,cur_floor, K)
        ups = np.array(ups).astype(np.int32)
        downs = np.array(downs).astype(np.int32)

        
        
        if len(ceiling)>0: 
            minuy = min(np.min(ups[:, 1]) - 10, -1)
            cv2.fillPoly(img=segmentation, pts=np.array(
            [[[ups[0, 0], minuy], *ups, [ups[-1, 0], minuy]]]), color=0)
        if len(floor)>0: 
            maxdy = max(np.max(downs[:, 1]) + 10, h+1)
            cv2.fillPoly(img=segmentation, pts=np.array(
                [[[downs[0, 0], maxdy], *downs, [downs[-1, 0], maxdy]]]), color=1)

        assert len(ups) == len(pwalls_output) + 1
        j = -1
        for i in range(len(ups)-1):
            u0 = ups[i]
            u1 = ups[i+1]
            
            if pwalls_output[i] is None:
                continue
            d0 = downs[i]
            d1 = downs[i+1]

            j = j + 1

            cv2.fillPoly(img=segmentation, pts=np.array(
                [[u0, d0, d1, u1]]), color=j+2)


        pwinverdepth = np.ones_like(segmentation) * 1e5
        # pixelwise
        
        depth = np.ones_like(segmentation) * 1e5
        labels = np.unique(segmentation)
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
                param = cur_ceiling
            elif label == 1:
                param = cur_floor
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
        if vis:
            plt.imshow(depth)
            plt.figure()
            plt.imshow(depth_ori)
            plt.figure()
            plt.imshow(segs)
            plt.figure()
            plt.imshow(segmentation)
            plt.show()
        depths.append(depth)
        segemantations.append(segmentation)
         
    
        image_iou, image_pe, merror_edge, rmse, us_rmse= evaluate(segs,cv2.resize(depth_ori, (640,360)), segmentation,depth, True, False)
        metric_results.append([image_iou, image_pe, merror_edge, rmse, us_rmse])


    

    if save:
        with open(os.path.join(filedir, 'metric_results.txt'), 'w') as file:
            for result in metric_results:
                file.write(str(result) + '\n')
            file.write(f"precision: {precision}, recall: {recall}\n")
        # for i,depth in enumerate(depths):
        #     depth_input_uint16 = (depth * 1000).astype(np.uint16)  # Scale depth to millimeters and convert to uint16
        #     depth_filename = os.path.join(filedir, f"depth_pred_{i}.png")
        #     cv2.imwrite(depth_filename, depth_input_uint16)  # Save depth image as PNG
        for i,seg in enumerate(segemantations):
            # cv2.imwrite(os.path.join(filedir, f"seg_pred_{i}.png"), seg)

            plt.imshow(seg)
            plt.savefig(os.path.join(filedir, f"seg_pred_vis_{i}.png"))
            plt.close()
            

    return metric_results


