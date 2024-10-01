import os
import cv2
import copy
import numpy as np
import json
import itertools
import trimesh

np.set_printoptions(suppress=True)
from sklearn.metrics import pairwise_distances
# from joblib import Parallel, delayed 

import matplotlib.pyplot as plt
import matplotlib
import os

def get_padded_bbx(seg, pad=10):
    ########## Seg returned from read_segmentation
    img_size = seg.shape

    seg1 = np.sum(seg, axis=0)
    seg2 = np.sum(seg, axis=1)

    cmin = np.where(seg1 > 0)[0][0]
    rmin = np.where(seg2 > 0)[0][0]

    cmax = np.where(seg1 > 0)[0][-1]
    rmax = np.where(seg2 > 0)[0][-1]

    rmin = max(rmin-pad, 0)
    rmax = min(rmax+pad, img_size[0])

    cmin = max(cmin-pad, 0)
    cmax = min(cmax+pad, img_size[1])

    return rmin, rmax, cmin, cmax

def get_paths(DATA_ROOT, scene, instance, frame, ext='png'):
    '''
        returns list of dicts, each containing fields

        depth_path
        img_path
        seg_path
        meta_path
    '''
    depth_path = os.path.join(DATA_ROOT, scene, "depth_NPZ", "{:04d}.npz".format(frame))
    img_path = os.path.join(DATA_ROOT, scene, "RGB", "{:04d}.".format(frame) + ext)
    seg_path = os.path.join(DATA_ROOT, scene, "segmentations", instance, "{}_{:04d}.".format(instance, frame) + ext)
    meta_path = os.path.join(DATA_ROOT, scene, "metadata_updated.json")
    try:
        assert os.path.exists(depth_path)
        assert os.path.exists(seg_path)
        assert os.path.exists(img_path)
        assert os.path.exists(meta_path)
    except:
        print('ERROR')
        print(depth_path)


    dct = dict(
        depth_path=depth_path,
        img_path=img_path,
        seg_path=seg_path,
        meta_path=meta_path)

    return dct


def get_pixel_grid(H, W):
    ### creating pixel points
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.array([x.flatten(), y.flatten()]).T

    x_co = coords[:, 0]
    y_co = coords[:, 1]
    z_co = np.ones_like(x_co)

    im_pixel_pts = np.stack([x_co, y_co, z_co]).T
    return im_pixel_pts


def read_segmentation(fpath):

    try:
        seg = cv2.imread(fpath)[:, :, 0]
    except:
        print(fpath)
    seg[seg <= 50] = 0.0
    seg[seg >= 50] = 1.0

    return seg

def read_segmentation2(fpath):

    try:
        seg = cv2.imread(fpath)
    except:
        print('read', fpath)
    seg[seg <= 50] = 0.0
    seg[seg >= 50] = 1.0

    return seg


    


