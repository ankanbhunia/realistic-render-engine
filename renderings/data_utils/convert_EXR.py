import numpy as np
import OpenEXR as exr
import Imath
import matplotlib.pyplot as plt
import os
import argparse
from joblib import Parallel, delayed 

parser = argparse.ArgumentParser()                                                         
parser.add_argument('--data_path', type=str, help='path to dataset to overlay backgrounds')
args = parser.parse_args()                                                                 
                                                                                           
src_dataset_path = args.data_path                                                          

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
                                                                                           
exr_file_paths = []       
subdirs = [sd for sd in os.listdir(src_dataset_path) if not sd.endswith('npz') 
    if not sd.endswith('.npy') if sd != 'split']


for d in subdirs: ### Each scene
    spath = os.path.join(src_dataset_path, d, 'depth_exr')
    for frame in os.listdir(spath):
        exr_file_paths.append([os.path.join(spath, frame), d])


def readEXR(filename):
    """Read RGB + Depth data from EXR image file.
    Parameters
    ----------
    filename : str
        File path.
    Returns
    -------
    img : RGB image in float32 format.
    Z : Depth buffer in float3.
    """
    
    exrfile = exr.InputFile(filename)
    dw = exrfile.header()['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    channels = ['R', 'G', 'B']
    channelData = dict()
    
    for c in channels:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)
        
        channelData[c] = C
    
    
    img = np.concatenate([channelData[c][...,np.newaxis] for c in ['R', 'G', 'B']], axis=2)
     
    return img

def job(arg):
    filename, scn_id = arg
    
    target_dir = os.path.join(args.data_path, scn_id, 'depth_NPZ')

    num = filename.split('/')[-1].split('.')[0]
    target_path = os.path.join(target_dir, num+'.npz')
    
    try:
        make_dir(target_dir)

        img = readEXR(filename)
        img = img[:,:,0]
        np.savez_compressed(target_path, img=img.astype(np.float16))
        return 1
    except:
        print(arg, 'failure')
        return scn_id

results = Parallel(n_jobs=12, verbose=1, backend="multiprocessing")(map(delayed(job), exr_file_paths))