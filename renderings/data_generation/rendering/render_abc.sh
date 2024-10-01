#!/bin/bash

data_path='/data/DevLearning/ABC_meshes_20220719'
# output_path='/home/ant/data/odme/ABC_rendering_output_easier'
output_path='/home/ant/data/odme/ABC_rendering_output_easier_amodal'
blender_path='/home/ant/Downloads/blender-2.93.7-linux-x64/blender'
scene_config_path='/data/DevLearning/odme/ABC_scene_configs/ABC_scene_configs_2023-02-13'
# scene_config_path='/data/DevLearning/odme/ABC_scene_configs/ABC_scene_configs_2022-07-21'

# scene_config_path='/home/ant/develop/CRIBpp_generic/common/ABC_scene_configs_2023-02-14'


CUDA_VISIBLE_DEVICES=2 python wrapper.py \
    --start=6000 \
    --end=8000 \
    --dataset_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --config_path=$scene_config_path \
    --dataset_type=ABC >&1 | tee datagen_abc.txt
