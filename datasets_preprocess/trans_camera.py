import numpy as np
import os
from tqdm import tqdm
import argparse

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

def process_scene(root):
    if not os.path.exists(root):
        raise ValueError(f"Dataset root path does not exist: {root}")
        
    for scene_id in tqdm(os.listdir(root), desc="Processing all scenes"):
        scene_path = os.path.join(root, f"{scene_id}")
        perspective_path = os.path.join(scene_path, "2D_rendering")
        for room_id in os.listdir(perspective_path):
            room_path = os.path.join(perspective_path, room_id, 'perspective', 'full')
            for position_id in os.listdir(room_path):
                position_path = os.path.join(room_path, position_id)
                camera_file = os.path.join(position_path, "camera_pose.txt")    
                camera_info = np.loadtxt(camera_file)
                pose, K = parse_camera_info(camera_info, 720, 1280)
                np.savez(os.path.join(position_path, "camera_pose.npz"), camera_pose=pose, camera_intrinsics=K)

def main():
    parser = argparse.ArgumentParser(description='Process camera poses from Structured3D dataset')
    parser.add_argument('--root', type=str, required=True,
                      help='Root directory of the Structured3D dataset')
    args = parser.parse_args()
    
    process_scene(args.root)

if __name__ == '__main__':
    main()

