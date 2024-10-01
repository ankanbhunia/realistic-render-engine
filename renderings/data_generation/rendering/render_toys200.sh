#!/bin/bash

# where you downloaded the blend file with all the objects
data_path='/home/ant/develop/CRIBpp_generic/objects2.blend'
# where you want the data to go
output_path='/data/DevLearning/odme/toys_rendering_output_single_test' 
# where the blender binary is
blender_path='/home/ant/Downloads/blender-2.91.2-linux64/blender'
# where all the scene configs files are
scene_config_path='/data/DevLearning/odme/toys200_scene_configs_single_test'

CUDA_VISIBLE_DEVICES=2 python wrapper.py \
    --start=1800 \
    --end=2000 \
    --dataset_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --config_path=$scene_config_path \
    --dataset_type=toys200 2>&1 | tee datagen_log_modelnet.txt
