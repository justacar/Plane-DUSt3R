import numpy as np
import json
import os
import matplotlib.pyplot as plt
from utils import *

def plane_merge(pointmaps, poses, non_cuboid_results, vis = False, save = False, save_dir = None):
    pose0 = poses[0]
    assert len(poses) == len(pointmaps)


    sum_ceiling_pparam = np.zeros(4)
    sum_floor_pparam = np.zeros(4)
    ceiling_count = 0
    floor_count = 0
    for i in range(len(non_cuboid_results)):
        trans = np.linalg.inv(pose0) @ poses[i]
        if len(non_cuboid_results[str(i)]["pceiling"]) > 0:
            sum_ceiling_pparam +=  non_cuboid_results[str(i)]["pceiling"] @ np.linalg.inv(trans)
            ceiling_count += 1
        if len(non_cuboid_results[str(i)]["pfloor"]) > 0:
            sum_floor_pparam += non_cuboid_results[str(i)]["pfloor"] @ np.linalg.inv(trans)
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

    xz_trans = transformation_matrix_to_align_xzplane(horizontal_pparam[:3])

    images_num = len(pointmaps)
    planes = []
    planeinfo_list = [[] for _ in range(images_num)]
    wall_relationship = [[] for _ in range(images_num)]
    for i in range(images_num):
        
        trans = np.linalg.inv(pose0) @ poses[i]
        
        # trans = np.eye(4)
        h,w,_ = pointmaps[i].shape

        points_homogeneous = np.hstack((pointmaps[i].reshape(-1, 3), np.ones((pointmaps[i].reshape(-1, 3).shape[0], 1))))

        pointmap_trans = (trans @ points_homogeneous.T).T

        pointmap_trans = pointmap_trans[:,:3].reshape(h,w,3)

        pwalls = non_cuboid_results[str(i)]["pwalls"]
        walls = non_cuboid_results[str(i)]["walls"]
        relations = non_cuboid_results[str(i)]["relations"]
        wall_relationship[i] = relations
        assert len(pwalls) == len(walls)

        if len(pwalls) == 1:
            pwall = pwalls[0] @ np.linalg.inv(trans)
            x1,y1,x2,y2 = map(int,walls[0][:4])
            center_3d = pointmap_trans[int((y1+y2)/2), int((x1+x2)/2)]
            planebox = pointmap_trans[y1:y2,x1:x2]
            normal = project_vector_to_plane(pwall[:3], horizontal_pparam[:3])
            plane_param = np.concatenate([normal, [pwall[3]]])
            longest_distance_pos_point, longest_distance_neg_point = longest_distance_to_center_under_threshold(planebox, center_3d,pwall, plane_param,xz_trans, 0.1) #TODO thresh
            pos_point_2d = (xz_trans[:3,:3] @ longest_distance_pos_point)[[0,2]]
            neg_point_2d = (xz_trans[:3,:3] @ longest_distance_neg_point)[[0,2]]
            center_2d = (xz_trans[:3,:3] @ center_3d)[[0,2]]

            curplane = Plane()
            curplane.pparam = pwall
            curplane.left_endpoint = longest_distance_pos_point
            curplane.right_endpoint = longest_distance_neg_point
            curplane.plane_center_2d = center_2d
            curplane.image_id = i
            curplane.line_segment = [pos_point_2d, neg_point_2d] #TODO
            planes.append(curplane)
            planeinfo_list[i].append(curplane)
        else:
        
            for j in range(len(pwalls)-1):

                pwall1 = pwalls[j] @ np.linalg.inv(trans)
                pwall2 = pwalls[j+1] @ np.linalg.inv(trans)
    

                x1,y1,x2,y2 = map(int,walls[j][:4])
                center_3d_1 = pointmap_trans[int((y1+y2)/2), int((x1+x2)/2)]
                planebox_1 = pointmap_trans[y1:y2,x1:x2]
                normal_1 = project_vector_to_plane(pwall1[:3], horizontal_pparam[:3])
                plane_param_1 = np.concatenate([normal_1, [pwall1[3]]])
                longest_distance_pos_point_1, longest_distance_neg_point_1 = longest_distance_to_center_under_threshold(planebox_1, center_3d_1,pwall1, plane_param_1,xz_trans, 0.1) #TODO thresh
                pos_point_2d_1 = (xz_trans[:3,:3] @ longest_distance_pos_point_1)[[0,2]]
                neg_point_2d_1 = (xz_trans[:3,:3] @ longest_distance_neg_point_1)[[0,2]]
                center_2d_1 = (xz_trans[:3,:3] @ center_3d_1)[[0,2]]

                x1,y1,x2,y2 = map(int,walls[j+1][:4])
                center_3d_2 = pointmap_trans[int((y1+y2)/2), int((x1+x2)/2)]
                planebox_2 = pointmap_trans[y1:y2,x1:x2]
                normal_2 = project_vector_to_plane(pwall2[:3], horizontal_pparam[:3])
                plane_param_2 = np.concatenate([normal_2, [pwall2[3]]])
                longest_distance_pos_point_2, longest_distance_neg_point_2 = longest_distance_to_center_under_threshold(planebox_2, center_3d_2,pwall2, plane_param_2,xz_trans, 0.1) #TODO thresh
                pos_point_2d_2 = (xz_trans[:3,:3] @ longest_distance_pos_point_2)[[0,2]]
                neg_point_2d_2 = (xz_trans[:3,:3] @ longest_distance_neg_point_2)[[0,2]]
                center_2d_2 = (xz_trans[:3,:3] @ center_3d_2)[[0,2]]


                if relations[j]:
                    line_point, line_dir = calculate_intersection_line(pwall1, pwall2)
                    line_point_2d = (xz_trans[:3,:3] @ line_point)[[0,2]]
                
                if j == 0:
                    curplane = Plane()
                    curplane.pparam = pwall1
                    curplane.left_endpoint = longest_distance_pos_point_1
                    curplane.right_endpoint = longest_distance_neg_point_1
                    curplane.plane_center_2d = center_2d_1
                    curplane.image_id = i
                    if relations[j]:
                        if np.linalg.norm(pos_point_2d_1-line_point_2d)<np.linalg.norm(neg_point_2d_1-line_point_2d):
                            curplane.line_segment = [neg_point_2d_1, line_point_2d]             
                        else:
                            curplane.line_segment = [pos_point_2d_1, line_point_2d]
                    else:
                        curplane.line_segment = [pos_point_2d_1, neg_point_2d_1] #TODO
                    planes.append(curplane)
                    planeinfo_list[i].append(curplane)
                    pre = curplane
                curplane = Plane()
                curplane.pparam = pwall2
                curplane.left_endpoint = longest_distance_pos_point_2
                curplane.right_endpoint = longest_distance_neg_point_2
                curplane.plane_center_2d = center_2d_2
                curplane.image_id = i
                if relations[j]:
                    pre.right = curplane
                    curplane.left = pre
                    if np.linalg.norm(pos_point_2d_2-line_point_2d)<np.linalg.norm(neg_point_2d_2-line_point_2d):
                        curplane.line_segment = [neg_point_2d_2, line_point_2d]
                        
                    else:
                        curplane.line_segment = [pos_point_2d_2, line_point_2d]
                else:
                    curplane.line_segment = [pos_point_2d_2, neg_point_2d_2] #TODO
                planes.append(curplane)
                planeinfo_list[i].append(curplane)
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


    if len(vertical_lines)>0:
        clusters_1 = cluster_lines(vertical_lines, horizontal_lines, 0.2, 0.2, 0.1)
    else:
        clusters_1 = []
    if len(horizontal_lines)>0:
        clusters_2 = cluster_lines(horizontal_lines, vertical_lines, 0.2, 0.2, 0.1)
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
        # print(" ")
        for line in node.lines:
            line.plane_pointer.global_id = i
            if line.diff_angle<min_diff_angle:
                min_diff_angle = line.diff_angle
                temp_pparam = line.plane_pointer.pparam
            
            # print(line.plane_pointer.pparam)
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
            # if line.is_vertical:
            #     plt.plot([-line.min_val,-line.max_val], [line.val,line.val],color=color_cycle[i%  len(color_cycle)])
            #     plt.text(-(line.min_val+line.max_val)/2,line.val, line.plane_pointer.global_id)
            # else:
            #     plt.plot([-line.val,-line.val],[line.min_val,line.max_val],color=color_cycle[i%  len(color_cycle)])
            #     plt.text(-line.val, (line.min_val+line.max_val)/2,line.plane_pointer.global_id)
                
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
    plt.axis('square')
    if vis:
        plt.show()
    node_data = {}
    nodes = []
    planes = {}
    for img_id in range(images_num):
        planes[str(img_id)] = [plane.global_id for plane in planeinfo_list[img_id]]


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
    node_data["planes"] = planes
    node_data["wall_relationship"] = wall_relationship

    if save:
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_dir, f"layout.png"))
        with open(os.path.join(save_dir, 'node_data.json'), 'w') as f:
            json.dump(node_data, f, indent=4)
    plt.close()
    return node_data