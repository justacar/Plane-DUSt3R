import sys
import os
import os.path as path
import argparse
HERE_PATH = path.normpath(path.dirname(__file__))
DUST3R_REPO_PATH = path.normpath(path.join(HERE_PATH, 'MASt3R','mast3r'))
if path.isdir(DUST3R_REPO_PATH):
    # workaround for sibling import
    sys.path.insert(0, path.join(HERE_PATH,  'MASt3R'))


import copy
import torch
import numpy as np  
from MASt3R.dust3r_extract import dust3r_extract
from dust3r.model import AsymmetricCroCo3DStereo
from NonCuboidRoom.plane_detection import extract_plane
from NonCuboidRoom.noncuboid.models import Detector
from easydict import EasyDict
import yaml
from plane_merge_planedust3r import plane_merge
import open3d as o3d

from PIL import Image

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #     R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def get_png_files(directory):
    png_files = []
    for file in os.listdir(directory):
        if  file.lower().endswith('.jpg') or file.lower().endswith('.png'): #file.lower().endswith('.png') or
            png_files.append(os.path.join(directory, file))
    return png_files



def find_intersect_point(plane1,plane2, plane3):
    A = np.array([plane1[:3], plane2[:3], plane3[:3]])
    b = np.array([-plane1[3], -plane2[3], -plane3[3]])
    # find the point where three planes intersect
    point_on_line = np.linalg.solve(A, b)
    return point_on_line

def project_point_to_intersection_line(plane1, plane2, point):
    """
    Project a 3D point onto the line of intersection of two planes.

    Args:
    plane1 (np.array): The first plane parameters as [nx, ny, nz, d], where n is the normal vector and d is the offset.
    plane2 (np.array): The second plane parameters as [nx, ny, nz, d].
    point (np.array): The 3D point to project as [x, y, z].

    Returns:
    np.array: The projected point on the line of intersection.
    """
    # Extract normal vectors and offsets from plane parameters
    n1, d1 = plane1[:3], plane1[3]
    n2, d2 = plane2[:3], plane2[3]

    # Calculate the direction vector of the line of intersection (cross product of normals)
    line_direction = np.cross(n1, n2)

    # To find a point on the line, solve the system of equations given by the plane equations
    # We can set one coordinate (z in this case) to zero to simplify solving
    A = np.array([n1, n2, line_direction])
    b = np.array([-d1, -d2, 0])

    # Solve for a point on the line
    try:
        line_point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # If the planes are parallel or coincident, return None
        return None

    # Project the given point onto the line
    # Calculate vector from point on line to the given point
    point_vector = point - line_point

    # Project this vector onto the line direction
    projection_length = np.dot(point_vector, line_direction) / np.dot(line_direction, line_direction)
    projected_point = line_point + projection_length * line_direction

    return projected_point




# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='3D Room Reconstruction')
    parser.add_argument('--dust3r_model', type=str, required=True,
                        help='Path to dust3r model checkpoint')
    parser.add_argument('--noncuboid_model', type=str, required=True,
                        help='Path to noncuboid model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images') # the size of all images should be the same
    parser.add_argument('--threshold', type=float, nargs=4, default=[0.35, 0.25, 0.25, 0.3],
                        help='Threshold values (4 float numbers)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    metric_flag = False
    dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(args.dust3r_model).to(args.device)

    noncuboid_model = Detector()
    state_dict = torch.load(args.noncuboid_model, map_location=torch.device(args.device))
    noncuboid_model.load_state_dict(state_dict)
    
    cfg_path = "NonCuboidRoom/cfg.yaml"
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
        cfg = EasyDict(config)

    # Get image list from input directory
    image_list = get_png_files(args.input_dir)

    dust3r_output = dust3r_extract(image_list, dust3r_model, metric=metric_flag)
    dust3r_image_size = (dust3r_output["pts3d"][0].shape[1], dust3r_output["pts3d"][0].shape[0])

    plane_detection = extract_plane(image_list, noncuboid_model, cfg, 
                                  threshold=tuple(args.threshold))


    node_info = plane_merge(dust3r_output, plane_detection, metric=metric_flag,
                          dust3r_image_size=dust3r_image_size, vis=True)

    pts3d = dust3r_output["pts3d"]


    floor = node_info["floor_pparam"]
    ceiling = node_info["ceiling_pparam"]

    if len(ceiling) == 0 and len(floor) != 0:
        ceiling = copy.deepcopy(floor)
        ceiling[1] = -ceiling[1]
    if len(floor) == 0 and len(ceiling) != 0:
        floor = copy.deepcopy(ceiling)
        floor[1] = -floor[1]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Viewer', width=800, height=600)
    
    for i, node in enumerate(node_info["global_plane_info"]):
        if node["pre"] is not None and node["next"] is not None:
            left_node_pparam = node_info["global_plane_info"][node["pre"]]["pparam"]
            right_node_pparam = node_info["global_plane_info"][node["next"]]["pparam"]
            node_pparam = node["pparam"]
            left_up_point = find_intersect_point(left_node_pparam,node_pparam,ceiling)
            right_up_point = find_intersect_point(right_node_pparam,node_pparam,ceiling)
            left_down_point = find_intersect_point(left_node_pparam,node_pparam,floor)
            right_down_point = find_intersect_point(right_node_pparam,node_pparam,floor)
        elif node["pre"] is None and node["next"] is not None:
            pparm0 = node_info["global_plane_info"][node["next"]]["pparam"]
            pparm1 = node["pparam"]
            right_up_point = find_intersect_point(pparm0,pparm1,ceiling)
            right_down_point = find_intersect_point(pparm0,pparm1,floor)
            left_up_point = project_point_to_intersection_line(ceiling,pparm1,node["left_endpoint"])
            left_down_point = project_point_to_intersection_line(floor,pparm1,node["left_endpoint"])
        elif node["pre"] is not None and node["next"] is None:
            pparm0 = node_info["global_plane_info"][node["pre"]]["pparam"]
            pparm1 = node["pparam"]
            left_up_point = find_intersect_point(pparm0,pparm1,ceiling)
            left_down_point = find_intersect_point(pparm0,pparm1,floor)
            right_up_point = project_point_to_intersection_line(ceiling,pparm1,node["right_endpoint"])
            right_down_point = project_point_to_intersection_line(floor,pparm1,node["right_endpoint"])
        else:
            pparm1 = node["pparam"]
            left_up_point = project_point_to_intersection_line(ceiling,pparm1,node["left_endpoint"])
            left_down_point = project_point_to_intersection_line(floor,pparm1,node["left_endpoint"])
            right_up_point = project_point_to_intersection_line(ceiling,pparm1,node["right_endpoint"])
            right_down_point = project_point_to_intersection_line(floor,pparm1,node["right_endpoint"])
        
        points = [left_up_point,right_up_point,left_down_point,right_down_point]
        points = np.array(points)
        
        line_mesh1 = LineMesh(points, [[0,1]], [1, 0, 0], radius=0.001)
        line_mesh1_geoms = line_mesh1.cylinder_segments
        line_mesh2 = LineMesh(points, [[0,2]], [1, 0, 0], radius=0.001)
        line_mesh2_geoms = line_mesh2.cylinder_segments
        line_mesh3 = LineMesh(points, [[1,3]], [1, 0, 0], radius=0.001)
        line_mesh3_geoms = line_mesh3.cylinder_segments
        line_mesh4 = LineMesh(points, [[2,3]], [1, 0, 0], radius=0.001)
        line_mesh4_geoms = line_mesh4.cylinder_segments
        
        vis.add_geometry(*line_mesh1_geoms)
        vis.add_geometry(*line_mesh2_geoms)
        vis.add_geometry(*line_mesh3_geoms)
        vis.add_geometry(*line_mesh4_geoms)

    points = []
    colors = []
    for i in range(len(image_list)):
        img = Image.open(image_list[i]).convert('RGB')
        img_resized = img.resize(dust3r_image_size)
        points.append(pts3d[i].reshape(-1, 3))
        colors.append(np.array(img_resized).reshape(-1, 3))
    
    points = np.concatenate(points,axis=0)
    colors = np.concatenate(colors,axis=0)
    points = points.reshape(-1,3)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1,3)/255)

    vis.add_geometry(point_cloud)
    vis.run()
    vis.destroy_window()




if __name__ == "__main__":
    main()

