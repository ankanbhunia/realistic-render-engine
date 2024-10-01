#!/bin/bash

data_path='/disk/nfs/gazinasvolume1/datasets/ShapeNetCore.v2'
output_path='shapenet_test'
blender_path='/disk/nfs/gazinasvolume2/s2514643/MOAD/LSME/data_generation/blender-2.93.18-linux-x64/blender'
scene_config_path='/disk/nfs/gazinasvolume2/s2514643/MOAD/LSME/data_generation/common/shapenet_scene_configs_test'
CUDA_VISIBLE_DEVICES=1
CAT_ID="03948459"

python data_generation/scene_config_generation/same_obj_create_scene_configs.py \
    --cat_id $CAT_ID\
    --N_FRAMES 20 \
    --N_SCENES 10 \
    --obj_size 0.4 \
    --OBJ_SCALE_MIN 0.35 \
    --OBJ_SCALE_MAX 0.45 \
    --CAM_RADIUS_MIN 0.7 \
    --CAM_RADIUS_MAX 0.9 \
    --CAM_HEIGHT_MIN 0.35 \
    --CAM_HEIGHT_MAX 0.65 \
    --MARGIN 0.5 

python data_generation/rendering/wrapper.py \
    --start=0 \
    --end=20 \
    --dataset_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --config_path=$scene_config_path \
    --dataset_type=shapenet 2>&1 | tee datagen_log_modelnet.txt