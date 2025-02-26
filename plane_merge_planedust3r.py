import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
from utils import *
# plt.switch_backend('agg')

def rescale_bbox(bbox_info, scale_x, scale_y):
    commons = np.array(bbox_info['commons'])
    ceiling = np.array(bbox_info['ceiling'])
    floor = np.array(bbox_info['floor'])
    plane = np.array(bbox_info['plane'])

    # rescale the plane detection
    if len(commons)>0:
        commons[:, [0,2]] = commons[:, [0,2]]*scale_x
        commons[:, [1,3]] = commons[:, [1,3]]*scale_y
    
    if len(plane)>0:
        plane[:, [0,2]] = plane[:, [0,2]]*scale_x
        plane[:, [1,3]] = plane[:, [1,3]]*scale_y

    if len(ceiling)>0:
        ceiling[:, [0,2]] = ceiling[:, [0,2]]*scale_x
        ceiling[:, [1,3]] = ceiling[:, [1,3]]*scale_y
    if len(floor)>0:
        floor[:, [0,2]] = floor[:, [0,2]]*scale_x
        floor[:, [1,3]] = floor[:, [1,3]]*scale_y
    return plane[:, :4] if len(plane)>0 else [], commons, ceiling, floor


def get_normal(pts3d, position, radius=0.1, max_nn=100):
    points = pts3d.reshape(-1,3)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn =min(250,max_nn)))
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(point_cloud, camera_location=[0,0,0])
    return np.asarray(point_cloud.normals)


def distance_point_to_line_3d(points, line_point, line_dir):
    """
    Calculate the distance from each point in a HxWx3 array to a line defined by a point and a direction vector.
    """
    # Vector from point on line to each point in question
    point_vectors = points - line_point
    # Projection of point_vectors onto the line direction
    line_dir_norm = np.linalg.norm(line_dir)
    proj_lengths = np.sum(point_vectors * line_dir, axis=-1) / line_dir_norm
    proj_vectors = np.outer(proj_lengths, line_dir / line_dir_norm).reshape(points.shape)
    # The shortest vectors from the points to the line are the orthogonal components of point_vectors to proj_vectors
    shortest_vectors = point_vectors - proj_vectors
    return np.linalg.norm(shortest_vectors, axis=-1)


def project_line_segment_onto_2d_line(point1, point2, line_coefficients):
    projected_point1 = project_points_onto_line_2d(point1[0], point1[1], line_coefficients)
    projected_point2 = project_points_onto_line_2d(point2[0], point2[1], line_coefficients)
    
    return projected_point1, projected_point2



def plane_merge(dust3r_output, plane_detection, vis = False, save = False, filedir = None, metric = False,image_size = (1280,720),dust3r_image_size = (512,288)):
    if metric:
        intersection_thresh = 0.1
        distance_thresh = 0.1
        x_thresh = 0.2
        margin_value = 0.1
    else:
        intersection_thresh = 0.01
        distance_thresh = 0.005
        x_thresh = 0.03
        margin_value = 0.01
        
    scale_x = dust3r_image_size[0]/image_size[0]  # dust3r output 512*288 image, the original size of image is 1280*720
    scale_y = dust3r_image_size[1]/image_size[1]  

    assert len(dust3r_output['pts3d']) == len(plane_detection)
    images_num = len(plane_detection)

    ceiling_count = 0
    floor_count = 0
    sum_ceiling_pparam = np.zeros(4)
    sum_floor_pparam = np.zeros(4)

    # calculate average floor and ceiling plane
    for img_id in range(images_num):
        # load bbox
        xyxy, commons, ceiling, floor = rescale_bbox(plane_detection[str(img_id)], scale_x, scale_y)
        if len(xyxy)==0:
            continue
        pts3d = dust3r_output['pts3d'][img_id]
        pose = dust3r_output['poses'][img_id]

        if len(ceiling)>0:
            left,up,right,down = map(int,ceiling[0][:4])
            centerx_ceiling, centery_ceiling = int((left+right)/2), int((up+down)/2)
            center_ceiling_3d = pts3d[centery_ceiling, centerx_ceiling]
            planebox = pts3d[up:down,left:right,:]
            ceiling_normals = get_normal(planebox, pose[:3,3],0.1, int((right-left)+(down-up)))
            index = (centery_ceiling-up) * (right-left) + (centerx_ceiling-left)
            ceiling_normal = ceiling_normals[index]
            ceiling_offset = -np.dot(ceiling_normal, center_ceiling_3d)
            ceiling_param= np.concatenate([ceiling_normal, [ceiling_offset]])
  
            if abs(ceiling_normal[1])>0.95:
                sum_ceiling_pparam += ceiling_param
                ceiling_count += 1

        if len(floor)>0:
            left,up,right,down = map(int,floor[0][:4])
            centerx_floor, centery_floor = int((left+right)/2), int((up+down)/2)
            center_floor_3d = pts3d[centery_floor, centerx_floor]
            planebox = pts3d[up:down,left:right,:]
            floor_normals = get_normal(planebox, pose[:3,3],0.1, int((right-left)+(down-up)))
            index = (centery_floor-up) * (right-left) + (centerx_floor-left)
            floor_normal = floor_normals[index]
            floor_offset = -np.dot(floor_normal, center_floor_3d)
            floor_param= np.concatenate([floor_normal, [floor_offset]])

            if abs(floor_normal[1])>0.95:
                sum_floor_pparam += floor_param
                floor_count += 1

    if ceiling_count==0:
        sum_ceiling_pparam = np.array([0,1,0,0])
    else:
        sum_ceiling_pparam /= ceiling_count
        sum_ceiling_pparam[:3] = sum_ceiling_pparam[:3] / np.linalg.norm(sum_ceiling_pparam[:3])
    if floor_count==0:
        sum_floor_pparam = np.array([0,-1,0,0])
    else:
        sum_floor_pparam /= floor_count
        sum_floor_pparam[:3] = sum_floor_pparam[:3] / np.linalg.norm(sum_floor_pparam[:3])
        
    
    horizontal_pparam = np.zeros(4)
    horizontal_pparam = (sum_floor_pparam+sum_ceiling_pparam)/2
    horizontal_pparam[1] = (sum_ceiling_pparam[1]-sum_floor_pparam[1])/2 # the direction of ceiling and floor is opposite
    horizontal_pparam[:3] = horizontal_pparam[:3] / np.linalg.norm(horizontal_pparam[:3])

    trans = transformation_matrix_to_align_xzplane(horizontal_pparam[:3])
    planeinfo_list = [[] for _ in range(images_num)]
    planes = []
    wall_relationship = [[] for _ in range(images_num)] # 1 for intersect, 0 for occlusion


    for img_id in range(images_num):
        plane_normals = []
        pts3d = dust3r_output['pts3d'][img_id]
        pose = dust3r_output['poses'][img_id]
        xyxy, commons, ceiling, floor = rescale_bbox(plane_detection[str(img_id)], scale_x, scale_y)
        if len(xyxy)==0:
            continue
        centerx = np.mean(xyxy[:, [0, 2]], axis=1)
        centery = np.mean(xyxy[:, [1, 3]], axis=1)

        for i in range(len(xyxy)):
            left,up,right,down = map(int,xyxy[i])
            planebox = pts3d[up:down,left:right,:]
            normals = get_normal(planebox, pose[:3,3],  0.1, int((right-left)+(down-up)))
            plane_normals.append(normals)
       
        
        if len(xyxy)==1:
            centerx,centery = int(centerx[0]), int(centery[0])
            left,up,right,down = map(int,xyxy[0])
            center_3d = pts3d[centery, centerx]
            index = (centery-up) * (right-left) + (centerx-left)
            normal_ori = plane_normals[0][index, :]
            normal = project_vector_to_plane(normal_ori, horizontal_pparam[:3])
            d_ori = -np.dot(normal_ori, center_3d)
            d = -np.dot(normal, center_3d)
            plane_param = np.concatenate([normal, [d]])
            plane_param_ori = np.concatenate([normal_ori, [d_ori]])
            wall_relationship[img_id].append(0)

            planebox = pts3d[up:down,left:right,:]
            longest_distance_pos_point, longest_distance_neg_point = longest_distance_to_center_under_threshold(planebox, center_3d,plane_param_ori, plane_param,trans, distance_thresh) #TODO thresh
            pos_point_2d = (trans[:3,:3] @ longest_distance_pos_point)[[0,2]]
            neg_point_2d = (trans[:3,:3] @ longest_distance_neg_point)[[0,2]]
            center_2d = (trans[:3,:3] @ center_3d)[[0,2]]
            curplane = Plane()
            curplane.pparam = plane_param_ori
            curplane.left_endpoint = longest_distance_pos_point
            curplane.right_endpoint = longest_distance_neg_point
            curplane.plane_center_2d = center_2d
            curplane.image_id = img_id

          
            curplane.line_segment = [pos_point_2d, neg_point_2d]
            
            planeinfo_list[img_id].append(curplane)
            planes.append(curplane)

        else:
            for i in range(len(xyxy)-1):  #TODO only one plane situation
                centerx1,centery1 = int(centerx[i]), int(centery[i])
                centerx2,centery2 = int(centerx[i+1]), int(centery[i+1])

                left1,up1,right1,down1 = map(int,xyxy[i])
                left2,up2,right2,down2 = map(int,xyxy[i+1])
    
                center_3d_1 = pts3d[centery1, centerx1]
                center_3d_2 =  pts3d[centery2, centerx2]

                index1 = (centery1-up1) * (right1-left1) + (centerx1-left1)
                index2 = (centery2-up2) * (right2-left2) + (centerx2-left2)

                normal1_ori = plane_normals[i][index1, :]
                normal2_ori = plane_normals[i+1][index2, :]

                normal1 = project_vector_to_plane(normal1_ori, horizontal_pparam[:3])
                normal2 = project_vector_to_plane(normal2_ori, horizontal_pparam[:3])

                # normal1 = normal1_ori
                # normal2 = normal2_ori

                d1_ori = -np.dot(normal1_ori, center_3d_1)
                d2_ori = -np.dot(normal2_ori, center_3d_2)

                d1 = -np.dot(normal1, center_3d_1)
                d2 = -np.dot(normal2, center_3d_2)

                plane_param_1 = np.concatenate([normal1, [d1]])
                plane_param_2 = np.concatenate([normal2, [d2]])


                plane_param_1_ori = np.concatenate([normal1_ori, [d1_ori]])
                plane_param_2_ori = np.concatenate([normal2_ori, [d2_ori]])


                # boudary of the potential intersection area of two bboxes
                lb, rb = int(commons[i][0]), int(commons[i][1])
                intersection = pts3d[:,lb:rb,:]
                line_point, line_dir = calculate_intersection_line(plane_param_1_ori, plane_param_2_ori)
                if line_point is not None and intersection.shape[1] > 0:
                    line_point_2d = (trans[:3,:3] @ line_point)[[0,2]]
                    distances = distance_point_to_line_3d(intersection, line_point, line_dir)
                    if distances.min()>intersection_thresh:
                        wall_relationship[img_id].append(0)
                    else:
                        wall_relationship[img_id].append(1)
                else:
                    wall_relationship[img_id].append(0)
                    distances = None

                x1,y1,x2,y2 = map(int,xyxy[i])
                if i == 0:
                    planebox = pts3d[y1:y2,x1:x2,:]
                    longest_distance_pos_point, longest_distance_neg_point = longest_distance_to_center_under_threshold(planebox, center_3d_1,plane_param_1_ori, plane_param_1,trans, distance_thresh) #TODO thresh
                    pos_point_2d = (trans[:3,:3] @ longest_distance_pos_point)[[0,2]]
                    neg_point_2d = (trans[:3,:3] @ longest_distance_neg_point)[[0,2]]
                    center_2d_1 = (trans[:3,:3] @ center_3d_1)[[0,2]]
                    curplane = Plane()
                    curplane.pparam = plane_param_1_ori
                    curplane.left_endpoint = longest_distance_pos_point
                    curplane.right_endpoint = longest_distance_neg_point
                    curplane.plane_center_2d = center_2d_1
                    curplane.image_id = img_id

                    if distances is not None and distances.min()<intersection_thresh:
                        if np.linalg.norm(pos_point_2d-line_point_2d)<np.linalg.norm(neg_point_2d-line_point_2d):
                            curplane.line_segment = [neg_point_2d, line_point_2d]
                        
                        else:
                            curplane.line_segment = [pos_point_2d, line_point_2d]
                        
                    else:
                        curplane.line_segment = [pos_point_2d, neg_point_2d]
                    
                    planeinfo_list[img_id].append(curplane)
                    planes.append(curplane)
                    pre = curplane
                    
                x1,y1,x2,y2 = xyxy[i+1]
                x1,y1 = int(x1), int(y1)
                x2,y2 = int(x2), int(y2)
        
                planebox = pts3d[y1:y2,x1:x2,:]

                longest_distance_pos_point, longest_distance_neg_point = longest_distance_to_center_under_threshold(planebox, center_3d_2, plane_param_2_ori, plane_param_2, trans, distance_thresh)

                pos_point_2d = (trans[:3,:3] @ longest_distance_pos_point)[[0,2]]
                neg_point_2d = (trans[:3,:3] @ longest_distance_neg_point)[[0,2]]
                center_2d_2 = (trans[:3,:3] @ center_3d_2)[[0,2]]

                curplane = Plane()
                curplane.pparam = plane_param_2_ori
                curplane.left_endpoint = longest_distance_pos_point
                curplane.right_endpoint = longest_distance_neg_point
                curplane.plane_center_2d = center_2d_2
                curplane.image_id = img_id
                if distances is not None and distances.min()<intersection_thresh:
                    pre.right = curplane
                    curplane.left = pre
                    if np.linalg.norm(pos_point_2d-line_point_2d)<np.linalg.norm(neg_point_2d-line_point_2d): 
                        curplane.line_segment = [neg_point_2d, line_point_2d]
                        
                    else:
                        curplane.line_segment = [pos_point_2d, line_point_2d]
                    
                else:
                    curplane.line_segment = [pos_point_2d, neg_point_2d]
                
                planeinfo_list[img_id].append(curplane)
                planes.append(curplane)
                pre = curplane
    
    angles = angles_to_axes(planes)

    # TODO more robust method eg: Median_absolute_deviation
    rotation_matrix = get_rotation_matrix(-np.median(angles))
    rotate_2d_segments(planes, rotation_matrix)


    vertical_lines = []
    horizontal_lines = []
    
    for i,plane in enumerate(planes):
        
        line_center = plane.plane_center_2d
        segment = plane.rotated_line_segment
        # plt.plot([segment[0][0],segment[1][0]],[segment[0][1],segment[1][1]],'r')
        # plt.axis('square')
        
        (point1,point2),is_vertical,diff_angle = determine_line_orientation_and_project(segment[0],segment[1],rotation_matrix @ line_center)
        if is_vertical:
            vertical_lines.append(OrthLine(point1,point2,is_vertical,diff_angle,plane))
        else:
            horizontal_lines.append(OrthLine(point1,point2,is_vertical,diff_angle,plane))
    # plt.show()
    #TODO
    # plt.figure()
    # for line in vertical_lines:
    #     (x1, y1), (x2, y2) = line.point1, line.point2
    #     plt.plot([x1,x2],[y1,y2],'g')
    # for line in horizontal_lines:
    #     (x1, y1), (x2, y2) = line.point1, line.point2
    #     plt.plot([x1,x2],[y1,y2],'r')
    # plt.axis('square')
   
    if len(vertical_lines)>0:
        clusters_1 = cluster_lines(vertical_lines, horizontal_lines, x_thresh, 0.2, margin_value)
    else:
        clusters_1 = []
    if len(horizontal_lines)>0:
        clusters_2 = cluster_lines(horizontal_lines, vertical_lines, x_thresh, 0.2, margin_value)
    else:
        clusters_2 = []

    clusters = clusters_1 + clusters_2
 
    chains = []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i,node in enumerate(clusters):
        chain_node = LinkedNode()
        sum_normal = np.array([0.0, 0.0, 0.0])
        sum_offset = 0.0
        min_diff_angle = 90
        temp_pparam = None
        line_count = 0
        for line in node.lines:
            line.plane_pointer.global_id = i
            if line.diff_angle<min_diff_angle:
                min_diff_angle = line.diff_angle
                temp_pparam = line.plane_pointer.pparam

            if line.diff_angle <5:

                line_count += 1
                sum_normal += line.plane_pointer.pparam[:3]
                sum_offset += line.plane_pointer.pparam[3]
        
            if line.is_vertical:
                plt.plot([line.val,line.val],[line.min_val,line.max_val],color=color_cycle[i%  len(color_cycle)])
                plt.text(line.val, (line.min_val+line.max_val)/2,line.plane_pointer.global_id)
            else:
                plt.plot([line.min_val,line.max_val],[line.val,line.val],color=color_cycle[i%  len(color_cycle)])
                plt.text((line.min_val+line.max_val)/2, line.val ,line.plane_pointer.global_id)
        if line_count == 0:
            chain_node.pparam = temp_pparam
        else:
            avg_normal = sum_normal / line_count
            avg_offset = sum_offset / line_count
            # Normalize the averaged normal vector to ensure it is a unit vector
            avg_normal /= np.linalg.norm(avg_normal)
            chain_node.pparam = np.concatenate([avg_normal, [avg_offset]])
        chain_node.global_id = i
        chains.append(chain_node)



    node_data = {}
    nodes = []
    planes_global = {}
    for img_id in range(images_num):
        planes_global[str(img_id)] = [plane.global_id for plane in planeinfo_list[img_id]]

    # pos_point_2d = (trans[:3,:3] @ longest_distance_pos_point)[[0,2]]
    # neg_point_2d = (trans[:3,:3] @ longest_distance_neg_point)[[0,2]]
    for i,node in enumerate(clusters):
        cur_left = None
        cur_right = None

        for line in node.lines:

            if line.plane_pointer.left:
                left_id = line.plane_pointer.left.global_id
                chains[i].pre = chains[left_id]
            if line.plane_pointer.right:
                right_id = line.plane_pointer.right.global_id
                chains[i].next = chains[right_id]

            # for truncated wall visualization
            left_point = line.plane_pointer.left_endpoint
            right_point = line.plane_pointer.right_endpoint
            if cur_left is not None and cur_right is not None:
                left_point_2d_cur = (trans[:3,:3] @ cur_left)[[0,2]]
                right_point_2d_cur = (trans[:3,:3] @ cur_right)[[0,2]]
                left_point_2d = (trans[:3,:3] @ left_point)[[0,2]]
                right_point_2d = (trans[:3,:3] @ right_point)[[0,2]]
                if np.linalg.norm(left_point_2d-right_point_2d_cur)>np.linalg.norm(left_point_2d_cur-right_point_2d_cur):
                    cur_left = left_point
                if np.linalg.norm(right_point_2d-left_point_2d_cur)>np.linalg.norm(right_point_2d_cur-left_point_2d_cur):
                    cur_right = right_point
            else:
                cur_left = left_point
                cur_right = right_point
        chains[i].left_endpoint = cur_left
        chains[i].right_endpoint = cur_right
    
    for i, node in enumerate(chains):
        
        node_info = {
            "index": node.global_id,
            "pparam": node.pparam.tolist(),  # Convert numpy array to list for JSON serialization
            "pre":node.pre.global_id if node.pre is not None else None,
            "next":node.next.global_id if node.next is not None else None,
            "left_endpoint":node.left_endpoint.tolist() if node.left_endpoint is not None else None,
            "right_endpoint":node.right_endpoint.tolist() if node.right_endpoint is not None else None,
        }
        nodes.append(node_info)
    node_data["global_plane_info"] = nodes
    node_data["floor_pparam"] = sum_floor_pparam.tolist() if floor_count>0 else []
    node_data["ceiling_pparam"] = sum_ceiling_pparam.tolist() if ceiling_count>0 else []
    node_data["planes"] = planes_global
    node_data["wall_relationship"] = wall_relationship


    plt.axis('square')

    if save:
        with open(os.path.join(filedir, 'node_data.json'), 'w') as f:
            json.dump(node_data, f, indent=4)
        # plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(filedir, 'layout.png'))
    if vis:
        plt.show() 
    plt.close()
    return node_data
