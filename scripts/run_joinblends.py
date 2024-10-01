import random, glob, os, shutil, time, argparse, tqdm
from anomaly_generators.utils import *

path = '/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files/'
out_path = '/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files_joined/'

blend_files = glob.glob(path+'/*/*.blend')

for blend_file_path in tqdm.tqdm(blend_files):

        try:
            out_blend_file_path = blend_file_path.replace(path, out_path)

            if os.path.isfile(out_blend_file_path):
                continue

            new_blend_folder_path = os.path.dirname(out_blend_file_path)
            os.makedirs(new_blend_folder_path, exist_ok = True)

            load_blend_file(blend_file_path)
            join_all_objects()
            bpy.ops.wm.save_as_mainfile(filepath=out_blend_file_path)

        except:
            print ('error')