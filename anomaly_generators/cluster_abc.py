import torch, glob, json
from PIL import Image
import torchvision.transforms as T
import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2

dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()

paths = glob.glob('/disk/scratch/s2514643/abc_data_processing/out_renders_2/*/')

patch_h, patch_w = 16, 16
transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

all_feats = []

for path in tqdm.tqdm(paths):

    image_tensor = torch.stack([transform(Image.open(i).convert('RGB')) for i in glob.glob(path+'/*')[:-1]]).cuda()

    feat = dinov2_vitg14(image_tensor).flatten().detach().cpu().numpy() 

    all_feats.append(feat)


X = np.stack(all_feats)

x_feats = torch.Tensor(X).cpu()

y = x_feats@x_feats.T
y_sim = y.argsort(descending = True)


my_dict = {'paths':paths, 'topk_indices':y_sim[:,:50].tolist()}

meta_str = json.dumps(my_dict, indent=True)

with open("/disk/scratch/s2514643/topk_indices.json", "w") as f:
    f.write(meta_str)

indices = y_sim[:,:50]

cv2.imwrite('im.png',np.concatenate([np.concatenate([cv2.imread(i) for i in glob.glob(paths[indices[1,0]]+'/*')], 1),
                     np.concatenate([cv2.imread(i) for i in glob.glob(paths[indices[1,1]]+'/*')], 1)],0))