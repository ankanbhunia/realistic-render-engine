import numpy as np
import os
import json

DATA_ROOT = "/home/ant/data/odme/ABC_rendering_output_easier"

all_scenes = os.listdir(DATA_ROOT)
all_scenes = [f for f in all_scenes if not f.endswith('npy') and not f.endswith('npz') and not f == 'split']
print(len(all_scenes))

os.makedirs(os.path.join(DATA_ROOT, 'split'), exist_ok=True)
split_path = os.path.join(DATA_ROOT, 'split', 'split2.json')
split_ratio = 0.8

train_scenes = np.random.choice(all_scenes, int(len(all_scenes)*split_ratio), replace=False)
print(len(train_scenes))

val_scenes = [s for s in all_scenes if s not in train_scenes]
print(len(val_scenes))

data = {'train': list(train_scenes), 'val': list(val_scenes)}

with open(split_path, 'w') as f:
	json.dump(data, f, indent=4)