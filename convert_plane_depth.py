import numpy as np
import json
import cv2
import os
import argparse
from shapely.geometry import Polygon
from tqdm import tqdm

K = np.array([[762, 0, 640], [0, -762, 360], [0, 0, 1]], dtype=np.float32)
K_inv = np.linalg.inv(K).astype(np.float32)

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

def inverdepth(param, K_inv, xy1map):
    n_d = param[:3] / np.clip(param[3], 1e-8, 1e8)  # meter n*1/d
    n_d = np.transpose(n_d, [1, 2, 0])
    inverdepth = -1 * np.sum(np.dot(n_d, K_inv) * xy1map, axis=2)
    return inverdepth

def process_scene(root):
    try:
        for scene_id in tqdm(os.listdir(root), desc="Processing all scenes"):
            scene_path = os.path.join(root, f"{scene_id}")
            perspective_path = os.path.join(scene_path, "2D_rendering")
            for room_id in os.listdir(perspective_path):
                room_path = os.path.join(perspective_path, room_id, 'perspective', 'full')
                for position_id in os.listdir(room_path):
                    position_path = os.path.join(room_path, position_id)
                    layout_file = os.path.join(position_path, "layout.json")
                    if os.path.exists(layout_file):
                        ratio_h = 2
                        inh, inw = 360, 640
                        pparams, segs = dataload(layout_file, ratio_h, inh, inw)

                        plane_params = np.zeros((4, 720, 1280), dtype=np.float32)
                        oseg = cv2.resize(segs, (1280,720), interpolation=cv2.INTER_NEAREST)
                        for i, param in enumerate(pparams):
                            param = np.array(param)
                            plane_params[:3, oseg == i] = param[:3, np.newaxis]  # normal
                            plane_params[3, oseg == i] = param[3]  # offset

                        x = np.arange(inw * 2)
                        y = np.arange(inh * 2)
                        xx, yy = np.meshgrid(x, y)
                        xymap = np.stack([xx, yy], axis=2).astype(np.float32)

                        ixymap = cv2.resize(xymap, (1280, 720), interpolation=cv2.INTER_LINEAR)
                        ixy1map = np.concatenate([ixymap, np.ones_like(ixymap[:, :, :1])], axis=-1).astype(np.float32)
                        inverdepth_input = inverdepth(plane_params, K_inv, ixy1map)
                        depth_input = 1/inverdepth_input
                        

                        depth_input_uint16 = (depth_input * 1000).astype(np.uint16)  # Scale depth to millimeters and convert to uint16
                        depth_filename = os.path.join(position_path, "plane_depth.png")
                        cv2.imwrite(depth_filename, depth_input_uint16)
    except Exception as e:
        print(f"Error occurred in file: {layout_file}")
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert plane layout to depth maps')
    parser.add_argument('--path', type=str, required=True, help='Path to the root directory containing scenes')
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        raise ValueError(f"The path {args.path} does not exist")
        
    process_scene(args.path)

if __name__ == "__main__":
    main()



