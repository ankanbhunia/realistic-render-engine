import random, glob, os, shutil, time, argparse, tqdm
from anomaly_generators.utils import *

path = '/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files/'
out_path = '/home/s2514643/MOAD-data-generation-latest/output_data/processed_blend_files/'

blend_files = glob.glob(path+'/*/*.blend')
filterd_ids = [i[:-1] for i in  open("good_ids.txt", "r").readlines()]

for blend_file_path in tqdm.tqdm(blend_files):

    if "-" not in os.path.basename(blend_file_path) or blend_file_path in filterd_ids:

        try:
            out_blend_file_path = blend_file_path.replace(path, out_path)

            if os.path.isfile(out_blend_file_path):
                continue

            new_blend_folder_path = os.path.dirname(out_blend_file_path)
            os.makedirs(new_blend_folder_path, exist_ok = True)
            re_pose_blend_file(blend_file_path, out_blend_file_path, plane_height = 0.05, rot_degree = [0,0,0])

            obj_path = out_blend_file_path.replace('.blend', '.obj')
            ply_path = out_blend_file_path.replace('.blend', '.ply')
            json_path = out_blend_file_path.replace('.blend', '.json')

            if os.path.isfile(obj_path) and os.path.isfile(ply_path):
                if not check_if_anomaly_visible(obj_path, ply_path):
                    os.remove(out_blend_file_path)
                    os.remove(obj_path)
                    os.remove(ply_path)
                    os.remove(json_path)

        except:
            print ('error')