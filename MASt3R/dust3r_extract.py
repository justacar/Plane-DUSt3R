import mast3r.utils.path_to_dust3r
import copy
import numpy as np
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy


def dust3r_extract(image_list, model, device = "cuda", save = False, filename = None, vis = False, metric = False,preset_poses = None,depth_flag = False):
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 150 #300
   
    single_image_flag = False
    # load_images can take a list of images or a directory
    images = load_images(image_list, size=512, verbose= False)
    image_num = len(images)
    if image_num == 1:
        images = [images[0], copy.deepcopy(images[0])]
        images[1]['idx'] = 1
        single_image_flag = True
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose= False)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose= False)
    if metric:
        scene.preset_metric()

    if preset_poses is not None:
        assert len(preset_poses) == image_num 
        if image_num == 1:
            scene.preset_pose([preset_poses[0]], [True])
        else:
            scene.preset_pose([preset_poses[i] for i in range(image_num)], [True for _ in range(image_num)])
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    focals = to_numpy(scene.get_focals())

    pts3d = to_numpy(pts3d)
    poses = to_numpy(poses)
    confidence_masks = to_numpy(confidence_masks)
    if single_image_flag:
        pts3d = pts3d[0][np.newaxis, ...]
        poses = poses[0][np.newaxis, ...]
        confidence_masks = confidence_masks[0][np.newaxis, ...]
    if save:
        np.savez(filename, pts3d=pts3d, poses=poses, confidence=confidence_masks)

    if vis:  
        scene.show()
        
    d = {}
    d["pts3d"] = pts3d
    d["poses"] = poses
    d["confidence"] = confidence_masks
    d["focals"] = focals
    return d

if __name__ == "__main__":
    import glob
    import os
    path = ""
    image_list = sorted(glob.glob(os.path.join(path, '*.png')))
    dust3r_extract(image_list, AsymmetricCroCo3DStereo.from_pretrained("").to("cuda"),vis=True)
    
