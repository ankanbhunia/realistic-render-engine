import random, glob, os, shutil, time, argparse, tqdm
from anomaly_generators.utils import *
import multiprocessing
from collections import Counter
from sklearn.cluster import OPTICS

path = "/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files/"
out_path = '/home/s2514643/MOAD-data-generation-latest/output_data/processed_blend_files/'

blend_objs_names = glob.glob(path+'/*/*.blend')
all_anomalies = [i for i in blend_objs_names if "-" in os.path.basename(i)]

modes = ['miss', 'scal', 'tran', 'rota']
all_types = [i for i in all_anomalies if (os.path.basename(i).split("-")[1] in modes) and (not os.path.isfile(i.replace('.blend', '.txt')))]

n_coms = []
while True:
    print (path)
    path = random.choice(all_types)
    if not os.path.isfile(path.replace('.blend', '.txt')):
        n_coms = optics_clustering(path)
        with open(path.replace('.blend', '.txt'), 'w+') as f:
            f.write(str(n_coms) + '\n')
    else:
        print ('skipping')
