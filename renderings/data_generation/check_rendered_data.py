import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import Counter

def check_data(arg):
    path = arg
    errors = []
    for p in path:
        count_files = len(os.listdir(p))
        if count_files > 7:
            errors.append(p)

    return errors

def get_statistics(arg):
    path = arg
    all_counts = []
    all_objects = []
    for p in path:
        seg_path = os.path.join(p, 'segmentations')
        seg_files = [f for f in os.listdir(seg_path) if f != "floor_object"]
        all_counts.append(len(seg_files))
        all_objects.append(seg_files)
    all_objects = np.concatenate(all_objects)
    return all_counts, list(all_objects)

data_path = '/home/ant/data/odme/ABC_rendering_output_easier'
all_files = os.listdir(data_path)
all_files = [f for f in all_files if not f.endswith('npz') if not f.endswith('npy')]
print(len(all_files))
all_file_paths = []
# num_files = np.array(["%05d"%(i) for i in range(50000)])
# missing_files = [a for a in num_files if a not in all_files]
# print(len(missing_files))
# np.savez(os.path.join(data_path, 'missing_files.npz'), missing=missing_files)
# import pdb; pdb.set_trace()


for f in all_files:
    file_path = os.path.join(data_path, f)
    all_file_paths.append(file_path)

print(len(all_file_paths))

num_split = 12
pool = Pool(num_split)

all_file_paths = np.asarray(all_file_paths)
print(len(all_file_paths))
all_paths_split = np.array_split(all_file_paths, num_split)


errors = pool.map(check_data, all_paths_split)
errors = np.concatenate(errors)
print('Errors: ', len(errors))
# print(errors)
# np.savez(os.path.join(data_path, 'error.npz'), errors=errors)


# ret = pool.map(get_statistics, all_paths_split)
# all_counts = [c[0] for c in ret]
# all_objects = [c[1] for c in ret]
# # import pdb; pdb.set_trace()
# all_counts = list(np.concatenate(all_counts))
# counter = Counter(all_counts)
# all_objects = list(np.concatenate(all_objects))
# counter_objects = Counter(all_objects)
# print(counter)
# print(counter_objects.most_common(50))
# print(len(counter_objects.keys()))
print('Done')
