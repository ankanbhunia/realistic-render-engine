import os
import numpy as np
import argparse
from joblib import Parallel, delayed 
from data_utils import read_segmentation
from PIL import Image

parser = argparse.ArgumentParser()                                                         
parser.add_argument('--data_path', type=str, help='path to dataset to overlay backgrounds')
args = parser.parse_args()  

src_dataset_path = args.data_path                                                          

rgb_file_paths = []
subdirs = [sd for sd in os.listdir(src_dataset_path) if not sd.endswith('npz')]


rgb_file_paths = [os.path.join(src_dataset_path, d, 'RGB') for d in subdirs if not d.endswith('npy') and not d.endswith('npz') if d != 'split']

print(len(rgb_file_paths))

def check_seg(spath):
    errors = []
    all_objects = [o for o in os.listdir(spath) if o != 'floor_object' and o!= 'merged_seg']
    for f in range(20):
        for obj in all_objects:
            obj_path = os.path.join(spath, obj, obj+'_%04d.png'%(f))
            try:
                obj_seg = read_segmentation(obj_path)
            except:
                errors.append(obj_path)
                print(obj_path)
    return errors

def merge_foreground(spath):
    all_objects = [o for o in os.listdir(spath) if o != 'floor_object' and o!= 'merged_seg']
    for f in range(20):
        merged_seg = np.zeros((224, 224))

        for obj in all_objects:
            obj_path = os.path.join(spath, obj, obj+'_%04d.png'%(f))
            obj_seg = read_segmentation(obj_path)
            merged_seg = merged_seg.astype(np.bool_) | obj_seg.astype(np.bool_)

        merged_seg = merged_seg.astype(np.float32)
        merged_seg = np.stack((merged_seg,)*3, -1)
        merged_seg = (merged_seg * 255.).astype(np.uint8)
        merged_seg = Image.fromarray(merged_seg)
        merged_seg.save(os.path.join(spath, 'merged_seg', 'merged_seg'+'_%04d.png'%(f)))

def check_rendering(spath):
    errors = []
    all_files = os.listdir(os.path.join(spath))
    for f in all_files:
        img1_path = os.path.join(spath, f)
        try:
            with open(img1_path, "rb") as f:
                img1 = Image.open(f).convert("RGB")
        except Exception:
            errors.append(img1_path)
            print(img1_path)
    return errors

def job(arg):
    spath = arg
    errors = check_rendering(spath)

    return errors

results = Parallel(n_jobs=12, verbose=1, backend="multiprocessing")(map(delayed(job), rgb_file_paths))
