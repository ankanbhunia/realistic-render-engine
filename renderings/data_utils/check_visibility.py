import os
import numpy as np
import argparse
from joblib import Parallel, delayed 
from data_utils import read_segmentation, read_segmentation2
import cv2
from PIL import Image
import shutil

parser = argparse.ArgumentParser()                                                         
parser.add_argument('--data_path', type=str, help='path to dataset to overlay backgrounds')
parser.add_argument('--shapenet', type=bool, default=False, help='path to dataset to overlay backgrounds')

args = parser.parse_args()  

src_dataset_path = args.data_path
shapenet = args.shapenet                                                          

seg_file_paths = []
subdirs = [sd for sd in os.listdir(src_dataset_path) if not sd.endswith('npz') if not sd == 'split']


seg_file_paths = [os.path.join(src_dataset_path, d) for d in subdirs if \
	not d.endswith('npy') and not d.endswith('npz')]
print(seg_file_paths)

def check_vis(spath1):
	spath = os.path.join(spath1, 'segmentations')
	all_objects = [o for o in os.listdir(spath) if o != 'floor_object' and o!= 'merged_seg']
	for obj in all_objects:
		not_vis = []
		for f in range(20):

			if args.shapenet:
				try:
					obj_path = os.path.join(spath, obj, obj.split('_')[1]+'_%04d.png'%(f))
				except:
					import shutil
					shutil.rmtree(os.path.join(spath, obj))
					break

			else:
				obj_path = os.path.join(spath, obj, obj+'_%04d.png'%(f))
			obj_seg = read_segmentation(obj_path)
			

			if np.sum(obj_seg) < 30:
				not_vis.append(f)
		if len(not_vis) < 19:
			np.savez(os.path.join(spath1, 'visibility', obj+'.npz'), not_visible=not_vis)
		if len(not_vis) >= 19:
			print(spath+ ' ' + obj)

		if len(not_vis) != 0:
			print(spath+ ' ' + obj)

			print(len(not_vis))

def job(arg):
	spath = arg
	try:
		if os.path.exists(os.path.join(spath, 'visibility')):
			shutil.rmtree(os.path.join(spath, 'visibility')) 
		os.makedirs(os.path.join(spath, 'visibility'), exist_ok=True)

		check_vis(spath)


		return 1
	except:
		print('fail')
		print(spath)
		print('-----')
		return spath

results = Parallel(n_jobs=12, verbose=1, backend="multiprocessing")(map(delayed(job), seg_file_paths))

