import random, glob, os, shutil, time, argparse, tqdm
from anomaly_generators.utils import *
import multiprocessing

path = "/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files/"
out_path = '/home/s2514643/MOAD-data-generation-latest/output_data/processed_glb_files/'

blend_objs_names = glob.glob(path+'/*/*.blend')

for blend_file_path in tqdm.tqdm(blend_objs_names):

    out_blend_file_path = blend_file_path.replace(path, out_path).replace(".blend", "glb")
    new_blend_folder_path = os.path.dirname(out_blend_file_path)
    os.makedirs(new_blend_folder_path, exist_ok = True)
    load_blend_file(blend_file_path)
    join_all_objects()
    bpy.ops.export_scene.gltf(filepath=out_blend_file_path)
