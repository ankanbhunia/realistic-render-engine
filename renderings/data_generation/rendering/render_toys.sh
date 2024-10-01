#!/bin/bash

data_path='/data/DevLearning/toys4k_blend_files'
output_path='/home/ant/develop/LSME/data_generation/toys_rendering_output'
blender_path='/home/ant/Downloads/blender-2.93.7-linux-x64/blender'
scene_config_path='/home/ant/develop/LSME/data_generation/common/toys_scene_configs'

python wrapper.py \
    --start=0 \
    --end=20 \
    --dataset_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --config_path=$scene_config_path \
    --dataset_type=toys 2>&1 | tee datagen_log_modelnet.txt
