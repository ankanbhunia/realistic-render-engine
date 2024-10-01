import imageio
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
# imageio.plugins.freeimage.download()

hdr_dirpath = 'common/assets/hdr'

all_hdrs = os.listdir(hdr_dirpath)

for hdr in all_hdrs:
    if not hdr.endswith('.hdr') or hdr.endswith('blurred.hdr') or hdr.replace('.hdr', '_blurred.hdr') in all_hdrs:
        continue


    hdr_path = os.path.join(hdr_dirpath, hdr)
    print(hdr_path)

    im = imageio.imread(hdr_path, format='HDR-FI')

    R, G, B  = im[:,:,0], im[:,:,1], im[:,:,2]
    bR = gaussian_filter(R, sigma=10)
    bG = gaussian_filter(G, sigma=10)
    bB = gaussian_filter(B, sigma=10)

    blurred = np.stack([bR, bG, bB],axis=2)

    hdr_out_path = hdr_path.replace('.hdr', '_blurred.hdr')
    imageio.imwrite(hdr_out_path, blurred, format='HDR-FI')