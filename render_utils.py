from PIL import Image
import os
import glob, json
from PIL import Image
import numpy as np
import cv2
#from bounding_box import bounding_box as bb

def create_gif(image_folder, output_gif_path, delay=100, loop=0):
    images = []
    
    # Read all images in the folder
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            img_path = os.path.join(image_folder, filename)
            images.append(Image.open(img_path))

    # Save as GIF
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=delay, loop=loop)

def create_gif_with_labels(output_render_path):

    output_gif_path = f'{output_render_path}/scene.mp4'
    os.makedirs(f'{output_render_path}/binary_labels', exist_ok = True)
    image_paths = sorted(glob.glob(f'{output_render_path}/RGB/*.png'))
    file_names = [os.path.basename(_) for _ in image_paths]
    bbox_dict = {i:[] for i in file_names}
    json_dict = json.load(open(f'{output_render_path}/scene3d.metadata.json', 'r'))
    obj_dict = {i['name']:'-' not in os.path.basename(i['path']) for i in json_dict['objects'] if i['path']} #maps obj_id to labels

    input_paths = [i['path'] for i in json_dict['objects']]
    labels = ['-' not in os.path.basename(i['path']) for i in json_dict['objects']]

    imgs = [np.array(Image.open(i)) for i in image_paths]
    seg_imgs_list = []  

    for obj_i in obj_dict:
        if obj_dict[obj_i] == 0:
            seg_paths = sorted(glob.glob(f'{output_render_path}/segmentations/{obj_i}/*.png'))
            seg_imgs = [Image.open(i).convert('L') for i in seg_paths]
            seg_imgs_list.append([np.array(j) for j in seg_imgs])
            for seg, im, name_ in zip(seg_imgs, imgs, file_names):
                binary_seg = seg.convert('L').point(lambda x: 0 if x < 127 else 255, '1')
                bbox_out = binary_seg.getbbox()
                if bbox_out:
                    x,y,x2,y2 = bbox_out
                    bbox_dict[name_].append([x,y,x2,y2])
        
                    cv2.rectangle(im, (x, y), (x2, y2), (0, 255, 0), 2)

    final_seg_labels = (np.array(seg_imgs_list).sum(0)>128).astype('uint8')*255

    #for seg_label, name_ in zip(final_seg_labels, file_names):
    #    cv2.imwrite(f'{output_render_path}/binary_labels/{name_}', seg_label)

    imgs = [cv2.cvtColor(image[:,:,:3], cv2.COLOR_BGR2RGB) for image in imgs]
    cv2.imwrite(output_gif_path.replace('.mp4', '.jpg'),np.concatenate(imgs[::4], 1)[:,:,:3] )

    cv2.imwrite("/disk/scratch/s2514643/abc_vis/"+output_gif_path.split('/')[-2]+'.jpg', np.concatenate(imgs[::4], 1)[:,:,:3] )

    output_video_path = output_gif_path
    video_resolution = imgs[0].shape[:2]
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, video_resolution)
    for image in imgs:
        video_writer.write(image)
    video_writer.release()

    scene_json_dict = {'blend_paths': input_paths, 'anomaly_labels': labels, 'bbox' : bbox_dict}

    with open(f'{output_render_path}/scene2d.metadata.json', 'w') as file:
        json.dump(scene_json_dict, file, indent = 2)


if __name__ == "__main__":

    input_folder = "/disk/nfs/gazinasvolume2/s2514643/MOAD/output_ehghe/RGB"
    output_gif = "output.gif"

    create_gif(input_folder, output_gif, delay=100, loop=0)