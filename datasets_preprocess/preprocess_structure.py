import os
import json
from itertools import combinations
from tqdm import tqdm
import argparse

def generate_dataset_json(root_path):
    training_set = []
    test_set = []
    
    for scene_id in tqdm(sorted(os.listdir(root_path))):
        scene_path = os.path.join(root_path, scene_id)
        if not scene_id.startswith('scene_'):
            continue
        scene_number = int(scene_id.split('_')[1])
        perspective_path = os.path.join(scene_path, "2D_rendering")
        
        for room_id in os.listdir(perspective_path):
            room_path = os.path.join(perspective_path, room_id, 'perspective', 'full')
            position_ids = os.listdir(room_path)
                
            position_combinations = list(combinations(position_ids, 2))
            for combo in position_combinations:
                item = {'sceneID': scene_id, 'roomID': room_id, 'positionID1': combo[0], 'positionID2': combo[1]}
                if scene_number <= 2999:
                    training_set.append(item)
                elif scene_number <= 3249:
                    test_set.append(item)

    # Uncomment the lines below to save the files
    with open(os.path.join(root_path, 'train_set.json'), 'w') as f:
        json.dump(training_set, f, indent=4)
    with open(os.path.join(root_path, 'test_set.json'), 'w') as f:
        json.dump(test_set, f, indent=4)
    
    # For now, just print one scene to check the output
    print("Training Set Sample (First Scene):", len(training_set))
    print("Test Set Sample (First Scene):", len(test_set))

def get_parser():
    parser = argparse.ArgumentParser(description='Generate dataset JSON files from Structured3D data')
    parser.add_argument('--root_path', type=str, required=True,
                       help='Root path to the Structured3D dataset')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    generate_dataset_json(args.root_path)



# import random

# # Reload the test set JSON and randomly select one item to print
# def reload_and_print_random_test_item():
#     test_set_path = os.path.join('/data2/mhf/DXL/hyx/strctured3d/Structured3D', 'test_set.json')
#     with open(test_set_path, 'r') as f:
#         test_set = json.load(f)
#     print(len(test_set))
#     print(type(test_set))
#     print(test_set[0])

# reload_and_print_random_test_item()
