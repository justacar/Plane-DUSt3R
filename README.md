
# Preprocess the dataset
Please first download the structured3d dataset and generate the plane depth images followed the guide of the main branch of the repository.

Split the dataset into train and test, the json files will be saved under the same directory of structured3d.

```
python datasets_preprocess/preprocess_structure.py --root_path /path/to/Structured3D
```
Transform the camera poses for easier data loading.

```
python datasets_preprocess/trans_camera.py --root /path/to/Structured3D
```



# Training

Following is the script we used to train the model.

```
torchrun --standalone my_train.py \​
    --train_dataset="Structured3d(split='train', ROOT='/path/to/Structured3D', aug_crop=256,mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) " \​
    --test_dataset="Structured3d(split='test', ROOT='/path/to/Structured3D', resolution=(512,384), seed=777) " \​
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \​
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \​
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12 ,freeze='encoder')" \​
    --pretrained="/path/to/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \​
    --lr=1e-04 --min_lr=1e-06 --warmup_epochs=2 --epochs=20 --batch_size=8 --accum_iter=2  \​
    --save_freq=1 --keep_freq=10 --eval_freq=1 \​
    --output_dir="/path/to/save/checkpoints"
```



