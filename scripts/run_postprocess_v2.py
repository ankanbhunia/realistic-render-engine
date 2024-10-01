import random, glob, os, shutil, time, argparse, tqdm
from anomaly_generators.utils import *
import multiprocessing
from collections import Counter

path = "/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files/"
out_path = '/home/s2514643/MOAD-data-generation-latest/output_data/processed_blend_files/'
simulate_numbers = 1

blend_objs_names = glob.glob(path+'/*/*.blend')
all_types = [os.path.basename(i).split("-")[1] for i in blend_objs_names if "-" in os.path.basename(i)]
Counter(all_types)

#for blend_file_path in tqdm.tqdm(blend_objs_names):

def _postprocess(blend_file_path):

    if os.path.isfile(blend_file_path.replace(".blend",".json")):
        return None

    try:
        is_normal = os.path.basename(blend_file_path).count('-') == 0
        out_blend_file_path = blend_file_path.replace(path, out_path)
        new_blend_folder_path = os.path.dirname(out_blend_file_path)
        os.makedirs(new_blend_folder_path, exist_ok = True)
        
        stable_poses = []

        for _ in range(simulate_numbers):
            
            rots = simulate_to_get_stable_pose(blend_file_path, plane_height = 0.05, rot_degree = 0.05)
            matrix = np.eye(4,4)
            matrix[:3,:3] = rots.T

            if not is_normal:

                pont_cloud_label = trimesh.load(blend_file_path.replace('.blend', '.ply'))
                pont_cloud_label.apply_transform(matrix.T)

                mesh_label = trimesh.load(blend_file_path.replace('.blend', '.obj'), force = 'mesh')
                mesh_label.apply_transform(matrix.T)

                is_visible = check_if_anomaly_visible(mesh_label, pont_cloud_label, min_vis_points = 10, \
                    num_views_check = 8, height = 5, radius = 5)

                stable_poses.append({'rot_matrix' : matrix.tolist(), 'is_visible':is_visible})
            
            else:

                stable_poses.append({'rot_matrix' : matrix.tolist()})

        if not is_normal:

            anomaly_mask = ((pont_cloud_label.colors[:,:3] == [0,0,0]).all(1))
            areas = [minimum_bounding_area(pont_cloud_label.vertices[anomaly_mask]), minimum_bounding_area(pont_cloud_label.vertices)]

            min_coords = np.min(pont_cloud_label.vertices[anomaly_mask], axis=0).tolist()
            max_coords = np.max(pont_cloud_label.vertices[anomaly_mask], axis=0).tolist()

            json_dict = {"stable_poses":stable_poses, "areas": areas, "bbox":[min_coords, max_coords]}

            load_blend_file(blend_file_path)
            join_all_objects()
            
            bpy.ops.wm.save_as_mainfile(filepath=out_blend_file_path)

        else:
            json_dict = {"stable_poses":stable_poses}
            shutil.copyfile(blend_file_path, out_blend_file_path)


        with open(out_blend_file_path.replace('.blend', '.json'), 'w') as json_file:
            json.dump(json_dict, json_file, indent=2)

    except:
        print ("error")


num_processes = 8
with multiprocessing.Pool(processes=num_processes) as pool:
    # Map the process_item function to the items in parallel
    results = pool.map(_postprocess, blend_objs_names)
