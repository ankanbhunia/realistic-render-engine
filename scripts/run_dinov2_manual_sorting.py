import subprocess, shutil, glob, time, cv2

import glob
import os
import torchvision.transforms as T
from PIL import Image
import numpy as np
import tqdm
path = "/disk/scratch/s2514643/rendered_objs_2/"
out_path = "/disk/scratch/s2514643/rendered_objs_concat/"

os.makedirs(out_path, exist_ok=True)

dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
patch_h, patch_w = 16, 16
transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

anomaly_paths = [i for i in glob.glob(path+'/*') if '-' in os.path.basename(i)]

for path in tqdm.tqdm(anomaly_paths[5780:]):

    try:
            
        im_view_as = [Image.open(i).convert('RGB') for i in glob.glob("".join(path.split('-')[:-2])+'/RGB/*')[:-1]]
        im_view_ns = [Image.open(i).convert('RGB') for i in glob.glob(path+'/RGB/*')[:-1]]

        image_tensor1 = torch.stack([transform(i) for i in im_view_as]).cuda()
        image_tensor2 = torch.stack([transform(i) for i in im_view_ns]).cuda()

        feat1 = dinov2_vitg14(image_tensor1).flatten().detach().cpu().numpy() 
        feat2 = dinov2_vitg14(image_tensor2).flatten().detach().cpu().numpy() 

        diff = ((feat1-feat2)**2).sum()

        im1 = np.concatenate([np.array(i) for i in im_view_as], 1)
        im2 = np.concatenate([np.array(i) for i in im_view_ns], 1)

        eroded_image = cv2.erode((np.abs(im1-im2).sum(-1)>10).astype('float32'), np.ones((3,3), np.uint8), iterations=2)
        
        out_img = np.concatenate([im1, im2, np.stack([eroded_image]*3, -1)*255], 0)
        out_path_name = os.path.join(out_path, str(int(diff))+'-'+os.path.basename(path)+'.png')

        cv2.imwrite(out_path_name, out_img)

        print (diff)
    
    except:

        print ('error')