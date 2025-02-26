import numpy as np

def normalize(vector):
    return vector / np.linalg.norm(vector)

def parse_camera_info(camera_info, height, width):
    """ extract intrinsic and extrinsic matrix
    """
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = -np.cross(W, U)

    rot = np.vstack((U, V, W))
    trans = camera_info[:3]

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    correction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    RT = np.eye(4)
    RT[:3,:3] = np.dot(correction, rot).T
    RT[:3,3] = trans/1000
    return RT, K



def transformation_matrix_to_align_xzplane(normal):
    """
    Compute the transformation matrix to align a given plane normal to the x-z plane.
    
    Args:
    normal (np.array): The normal vector of the plane.
    
    Returns:
    np.array: A 4x4 transformation matrix.
    """
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Define the target normal for the x-z plane (y-axis normal)
    target_normal = np.array([0, 1, 0])
    if normal[1] == 1:
        return np.eye(4)
    
    # Compute the rotation axis using cross product
    rotation_axis = np.cross(normal, target_normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Compute the angle between the normals
    angle = np.arccos(np.dot(normal, target_normal))
    
    # Compute the rotation matrix around the rotation axis by the angle
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    ux, uy, uz = rotation_axis
    
    # Rodrigues' rotation formula
    R = np.array([
        [cos_angle + ux**2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
        [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy**2 * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle],
        [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz**2 * (1 - cos_angle)]
    ])
    
    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    
    return transformation_matrix

def project_points_onto_line_2d(points, line_coefficients):
    """
    Project points onto a line defined by the line equation ax + by + c = 0 using numpy operations for efficiency.
    This function supports points being either a single point (px, py) or multiple points in a 2D array (Nx2) or a 3D array (HxWx2).

    Args:
    points (np.array): Either a tuple (px, py) or an Nx2 or HxWx2 numpy array where each row represents the coordinates (x, y) of a point.
    line_coefficients (np.array): Coefficients [a, b, c] of the line equation ax + by + c = 0.

    Returns:
    np.array: Projected points, either as a single point [x_proj, y_proj] or an Nx2 or HxWx2 array of projected points, matching the input shape.
    """
    a, b, c = line_coefficients
    if points.ndim == 1:  # Single point
        px, py = points
        d = a * px + b * py + c
        denom = a**2 + b**2
        x_proj = px - a * d / denom
        y_proj = py - b * d / denom
        return np.array([x_proj, y_proj])
    else:  # Multiple points
        d = a * points[..., 0] + b * points[..., 1] + c
        denom = a**2 + b**2
        x_proj = points[..., 0] - a * d / denom
        y_proj = points[..., 1] - b * d / denom
        return np.stack([x_proj, y_proj], axis=-1)


def longest_distance_to_center_under_threshold(points, center, plane, plane_horizon, trans, threshold):
    """
    Calculate the longest distance from points to a center point under a given threshold.
    """
    h, w, _ = points.shape

    plane_normal = plane[:3]
    plane_offset = plane[3]
    
    # Calculate point-to-plane distances
    point_to_plane_distances = np.abs(np.dot(points, plane_normal) + plane_offset) / np.linalg.norm(plane_normal)
    # import matplotlib.pyplot as plt
    # plt.imshow(point_to_plane_distances)
    # plt.show()
    valid_mask = point_to_plane_distances < threshold

    valid_points = points[valid_mask]
    
    assert len(valid_points) != 0

    trans_plane = trans @ plane_horizon
    line_coefficients = np.array([trans_plane[0], trans_plane[2], trans_plane[3]])

    points_2d = (trans[:3,:3] @ points.reshape(-1, 3).T).T[:,[0,2]]

    projected_point = project_points_onto_line_2d(points_2d, line_coefficients)

    center_2d = (trans[:3,:3] @ center)[[0,2]]
    point_vector = projected_point - center_2d

    
    x_indices = np.tile(np.arange(w), h).reshape(h, w)
    pos_mask = (x_indices < w / 2) & valid_mask
    neg_mask = (x_indices >= w / 2) & valid_mask


    point_to_center_distance = np.linalg.norm(point_vector, axis=-1)
    point_to_center_distance = point_to_center_distance.reshape(h, w)
  
    longest_distance_index_pos = np.argmax(point_to_center_distance[pos_mask])
    longest_distance_index_neg = np.argmax(point_to_center_distance[neg_mask])

    longest_distance_pos_point = points[pos_mask][longest_distance_index_pos]
    longest_distance_neg_point = points[neg_mask][longest_distance_index_neg]

    return longest_distance_pos_point, longest_distance_neg_point


def calculate_intersection_line(plane1, plane2):
    """
    Calculate the intersection line of two 3d planes given in normal and offset format.
    plane1 and plane2 are tuples (normal_vector, offset).
    """
    n1, d1 = plane1[:3], plane1[3]
    n2, d2 = plane2[:3], plane2[3]
    # Cross product of normals gives a direction vector for the line of intersection
    direction = np.cross(n1, n2)
    # To find a point on the line, solve the system of equations given by the plane equations
    # We can set one coordinate (z in this case) to zero to simplify solving
    A = np.array([n1, n2, direction])
    b = np.array([-d1, -d2, 0])
    # find the point where three planes intersect
    try:
        point_on_line = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None
    return point_on_line, direction


def project_vector_to_plane(vector, plane_normal):
    """
    Project a 3D vector onto a plane defined by its normal vector.
    
    Args:
    vector (np.array): A numpy array representing the 3D vector to be projected.
    plane_normal (np.array): A numpy array representing the normal vector of the plane.
    
    Returns:
    np.array: A numpy array representing the projected vector on the plane.
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Ensure the normal is a unit vector
    projection = vector - np.dot(vector, plane_normal) * plane_normal
    return projection

def angles_to_axes(planes):
    import numpy as np
    angles = []
    for plane in planes:
        (x1, y1), (x2, y2) = plane.line_segment
        # Calculate the angle of the line with respect to the horizontal axis
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        if -45<angle_deg<45:
            angles.append(angle_deg)
        elif 45<angle_deg<135:
            angles.append(angle_deg-90)
        elif -135<angle_deg<-45:
            angles.append(angle_deg+90)
        elif -180<angle_deg<-135:
            angles.append(angle_deg+180)
        elif -130<angle_deg<180:
            angles.append(angle_deg-180)
        else:
            angles.append(0)
        # Append the minimum angle to the list
       
    return angles

def determine_line_orientation_and_project(end_point1, end_point2, center):
    """
    Determine if the line formed by two endpoints is closer to vertical or horizontal,
    and project the endpoints onto the line y=center[1] if vertical, or x=center[0] if horizontal.
    Also, return a flag indicating if the line is vertical and the difference of angle with horizon or vertical.

    Args:
    end_point1 (tuple): The first endpoint of the line (x1, y1).
    end_point2 (tuple): The second endpoint of the line (x2, y2).
    center (tuple): The center point (cx, cy) used to determine the projection axis.

    Returns:
    tuple: The projected endpoints, a boolean flag (True if vertical, False if horizontal), and the angle difference.
    """
    x1, y1 = end_point1
    x2, y2 = end_point2
    cx, cy = center

    # Calculate the angle of the line with respect to the horizontal axis
    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1))
    angle_deg = np.degrees(angle)

    # Determine if the line is closer to vertical or horizontal
    if abs(y2 - y1) > abs(x2 - x1):
        # Line is closer to vertical, project onto y = cy
        angle_diff = 90 - angle_deg

        return ((cx, y1), (cx, y2)), True, angle_diff
    else:
        # Line is closer to horizontal, project onto x = cx
        angle_diff =  angle_deg
        return ((x1, cy), (x2, cy)), False, angle_diff


def get_rotation_matrix(angle_degrees):
    """
    Generate a 2D rotation matrix for a given angle in degrees.

    Args:
    angle_degrees (float): The angle in degrees for which to generate the rotation matrix.

    Returns:
    numpy.ndarray: A 2x2 rotation matrix.
    """
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return rotation_matrix


def rotate_2d_segments(planes, rotation_matrix):
    """
    Rotate 2D segments by a given angle.

    Args:
    segments (list of tuples): List of segments, where each segment is represented by ((x1, y1), (x2, y2)).
    angle_degrees (float): The angle by which to rotate the segments, in degrees.

    Returns:
    list of tuples: List of rotated segments.
    """

    # rotated_segments = []
    for plane in planes:
        (x1, y1), (x2, y2) = plane.line_segment
        point1 = np.array([x1, y1])
        point2 = np.array([x2, y2])
        rotated_point1 = rotation_matrix @ point1
        rotated_point2 = rotation_matrix @ point2
        # rotated_segments.append((rotated_point1, rotated_point2))
        plane.rotated_line_segment = (rotated_point1, rotated_point2)

    # return rotated_segments 


class OrthLine():
    def __init__(self,point1,point2,is_verticaled,diff_angle,plane):
        self.point1 = point1
        self.point2 = point2
        self.is_vertical = is_verticaled
        self.diff_angle = diff_angle
        self.global_id = None
        self.plane_pointer = plane
        self.pparam = None
        if self.is_vertical:
            assert point1[0] == point2[0]
            self.val = point1[0]
            self.min_val = min(point1[1],point2[1])
            self.max_val = max(point1[1],point2[1])
        else:
            assert point1[1] == point2[1]
            self.val = point1[1]
            self.min_val = min(point1[0],point2[0])
            self.max_val = max(point1[0],point2[0])

class Node(): 
    def __init__(self, line):
        self.lines = [line]
        self.min_val = line.min_val
        self.max_val = line.max_val
        self.centroid = line.val
        self.num = 1
        self.image_ids = []
    
    def get_centroid(self):
        if self.num == 0:
            return None
        else:
            return self.centroid/self.num
        
    def insert(self,line):
        self.lines.append(line)
        self.num += 1
        self.centroid += line.val
        self.min_val = min(self.min_val, line.val)
        self.max_val = max(self.max_val, line.val)
        self.image_ids.append(line.plane_pointer.image_id)

class Plane():
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.global_id = None
        self.pparam = None
        self.line_segment = None
        self.rotated_line_segment = None
        self.plane_center_2d = None
        self.image_id = None
        self.left_endpoint = None
        self.right_endpoint = None
       
class LinkedNode:
    def __init__(self):
        self.global_id = None
        self.pparm = None
        self.pre = None
        self.next = None
        self.left_endpoint = None
        self.right_endpoint = None

    def assign(self,value):
        self.pparm = value

    def __repr__(self):
        return f"<LinkedNode value:{self.value} pre:{self.pre.value if self.pre is not None else 'no pre'} next:{self.next.value if self.next is not None else 'no next'}>"

    def __str__(self):
        return f"From str method of Test: {self.value} pre:{self.pre.value if self.pre is not None else 'no pre'} next:{self.next.value if self.next is not None else 'no next'}"
      

def cluster_lines(vertical_lines, horizontal_lines, x_thresh, y_overlap_thresh, margin_value):
    # TODO if line from the same image, can not be in the same cluster
    def overlap(line,node):
        if line.max_val <= node.min_val or line.min_val >= node.max_val:
            return 0
        else:
            propotion = (min(line.max_val, node.max_val) - max(line.min_val, node.min_val))/(line.max_val - line.min_val)
            return propotion
        
    def is_intersect(v_line, node, h_line, margin_value):
        if v_line.val + margin_value <= h_line.min_val or v_line.val - margin_value >= h_line.max_val:
            return False
        up_boundary = min(v_line.max_val, node.max_val)
        down_boundary = max(v_line.min_val, node.min_val)
        if up_boundary<down_boundary:
            return up_boundary<h_line.val<down_boundary
        else:
            return down_boundary<h_line.val<up_boundary

    vertical_lines = sorted(vertical_lines, key=lambda x: x.val)
    node = Node(vertical_lines[0])
    node.image_ids.append(vertical_lines[0].plane_pointer.image_id)
    cluster_nodes = [node]
    
    for line in vertical_lines[1:]:
        merge_node_found = False
        for node in cluster_nodes:
            if line.plane_pointer.image_id in node.image_ids:
                continue
            if abs(line.val - node.get_centroid()) < x_thresh:
                overlap_ratio = overlap(line,node)
                if overlap_ratio >= y_overlap_thresh:
                    node.insert(line)
                    merge_node_found = True
                    break
                else:
                    intersect_line_found = False
                    for h_line in horizontal_lines:
                        if is_intersect(line, node, h_line, margin_value):
                            intersect_line_found = True
                            break
                    if not intersect_line_found:
                        node.insert(line)
                        merge_node_found = True
                        break
        if not merge_node_found:
            node = Node(line)
            node.image_ids.append(line.plane_pointer.image_id)
            cluster_nodes.append(node)
    return cluster_nodes
