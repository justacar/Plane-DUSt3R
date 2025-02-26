import copy
import os
import shutil
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.device import to_numpy
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def mast3r_extract(image_list, model,vis = False, save = False, filename = None):
    imgs = load_images(image_list, size=512, verbose=False)
    single_img_flag = False
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        image_list = [image_list[0], image_list[0] + '_2']
        single_img_flag = True
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)

    scene = sparse_global_alignment(image_list, pairs, "t",
                                    model, lr1=0.07, niter1=300, lr2=0.014, niter2=100, device="cuda",
                                    opt_depth='depth' in "refine+depth", shared_intrinsics=True,
                                    matching_conf_thr=5, verbose=False)
    # Delete the "t" folder created during optimization

    
    cams2world = to_numpy(scene.get_im_poses())
    if single_img_flag:
        cams2world = cams2world[0][np.newaxis, ...]
    if save:
        np.save(filename, cams2world)
    if vis:
        scene.show()
    if os.path.exists("t"):
        shutil.rmtree("t")
        # print("delete t")
    return cams2world



if __name__ == "__main__":
    import os
    import logging
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    LOG = logging.getLogger(__name__)

    logging.basicConfig(level=logging.INFO)

    weights_path = "checkpoint-best-metric.pth"
    model = AsymmetricMASt3R.from_pretrained(weights_path).to("cuda")
    



    root_path = ""
    save_path = ""

    

    with logging_redirect_tqdm():
        for scene_id in tqdm(sorted(os.listdir(root_path))):
        
            LOG.info(f"Processing scene {scene_id}")
            scene_path = os.path.join(root_path, scene_id)
            scene_number = scene_id.split('_')[1]
            perspective_path = os.path.join(scene_path, "2D_rendering")
            for room_id in os.listdir(perspective_path):
            
                room_path = os.path.join(perspective_path, room_id, 'perspective', 'full')
                position_ids = sorted(os.listdir(room_path))
                # if len(position_ids)<2:
                #     continue
                image_list = []
                for position_id in position_ids:
                    position_path = os.path.join(room_path, position_id)
                    image_list.append(os.path.join(position_path, 'rgb_rawlight.png'))

                result_dir = os.path.join(save_path,scene_number, room_id)
                os.makedirs(result_dir, exist_ok=True)
                mast3r_extract(image_list, model, save = True, filename=os.path.join(result_dir, "dust3r_output.npz"))

