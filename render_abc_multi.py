import subprocess, shutil, glob, time


from render_utils import *
from anomaly_generators.utils import convert_and_save_as_joined
from anomaly_generators.utils import *

import fcntl

def write_to_log(log_file, message):
    with open(log_file, 'a') as f:
        # Acquire an exclusive lock on the file
        fcntl.flock(f, fcntl.LOCK_EX)
        
        # Write to the log file
        f.write(message + '\n')
        f.flush()  # Flush the buffer to ensure the message is written immediately
        
        # Release the lock
        fcntl.flock(f, fcntl.LOCK_UN)

def read_from_log(log_file):
    with open(log_file, 'r') as f:
        # Acquire a shared lock on the file
        fcntl.flock(f, fcntl.LOCK_SH)
        
        # Read from the log file
        log_contents = f.readlines()
        
        # Release the lock
        fcntl.flock(f, fcntl.LOCK_UN)
    
    return log_contents


## min_multiplier -> {1,2} high means more space between objects
## radius_constant -> [0.02, 0.1] higher means higher camera distance
## if number of objects in a dataset is higher then above values need to be higher 


def render_scene_multi(obj_files, out_folder, fg_pbr_path, bg_pbr_path, env_hdr_path, min_multiplier=2, radius_constant=0.08, num_views = 20):

    def func_render_scene(obj_paths, dataset_name, output_path, pose_file_name, bg_pbr_path, fg_pbr_path, env_hdr_path, min_multiplier=2, radius_constant=0.08, num_views = 20):

        subprocess.call([os.path.abspath("renderings/data_generation/blender-2.93.18-linux-x64/blender"),
        "--background", os.path.abspath("renderings/data_generation/common/empty_scene.blend"), 
        "--python", os.path.abspath("renderings/data_generation/rendering/generate.py"), "--", "--dataset_type", 
        dataset_name, "--output_path", output_path, "--pose_file_name", pose_file_name,
        "--min_multiplier", str(min_multiplier), "--radius_constant", str(radius_constant),
        "--bg_pbr_path", str(bg_pbr_path), "--fg_pbr_path", str(fg_pbr_path), "--env_hdr_path", str(env_hdr_path),
        "--obj_paths_input",]+obj_paths+["--N_FRAMES", str(num_views)], \
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    temp_file_name = tempfile.mktemp()+'.npy'
    random_angle = [10 + np.random.rand()*10 for _ in range(3)]
    out_list = {}
    for obj_path in obj_files:
        rots = simulate_to_get_stable_pose(obj_path, 5, random_angle)
        matrix = np.eye(4,4)
        matrix[:3,:3] = rots.T
        out_list[os.path.basename(obj_path)] = matrix
    
    np.save(temp_file_name, out_list)

    ret_code = func_render_scene(obj_files, \
            dataset_name = "ABC", \
            output_path = out_folder, \
            num_views = num_views, \
            pose_file_name = temp_file_name,
            fg_pbr_path = fg_pbr_path,
            bg_pbr_path = bg_pbr_path,
            env_hdr_path = env_hdr_path)



def random_sample_one_texture_map_(ALL_MATERIALS_PATH, choose_from = [], empty_prob = 0.0):
    if random.random()>empty_prob:
        if len(choose_from)>0:
            return random.choice([i for i in ALL_MATERIALS_PATH if i.split('/')[-1] in choose_from])
        else:
            return random.choice(ALL_MATERIALS_PATH)
    else:
        return ''

def sample_one_shape_from_(ALL_SHAPES, choose_from = []):

    def _check_shape_size_(file_path, thres): #mb
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            if file_size_mb > thres:
                print(f"File is {file_size_mb:.2f} MB, above 20 MB, discarding it.")
                return False
            else: 
                return True
        else:
            print("File does not exist.")
            return False
    
    while True:
        try:
            if len(choose_from)>0:
                file_path = random.choice(sum([[j for j in ALL_SHAPES if i in j] for i in choose_from], []))
                return file_path
            else:
                file_path = random.choice(ALL_SHAPES)

            if _check_shape_size_(file_path, thres = 20):
                return file_path
        except:
            return None

if __name__ == "__main__":

    ALL_MATERIALS_PATH = glob.glob('/disk/nfs/gazinasvolume2/s2514643/MatSynth/*/*/*')
    ALL_ENVS_PATH = glob.glob('/disk/nfs/gazinasvolume2/s2514643/look3d-o3-rendering/data/env_files/*hdr')
    ALL_NORMAL_SHAPES = glob.glob('/disk/nfs/gazinasvolume2/s2514643/look3d-o3-rendering/data/abc_dataset/*.obj')
    ALL_ANOMALY_SHAPES = glob.glob('/disk/nfs/gazinasvolume2/s2514643/look3d-o3-rendering/data/abc_anomaly/*/*.obj')
    good_ids = ['-'.join(i[:-5].split('-')[1:]) for i in open('/disk/nfs/gazinasvolume2/s2514643/look3d-o3-rendering/data/good_ids.txt','r').readlines()]

    OUT_DIR = '/disk/nfs/gazinasvolume2/s2514643/look3d-o3-rendering/data/abc_rendering_reuslts_multi/'
    os.makedirs(OUT_DIR, exist_ok=True)

    anomaly_obj_file_path = sample_one_shape_from_(ALL_ANOMALY_SHAPES)
    anomaly_shape_id = anomaly_obj_file_path.split('/')[-2]
    normal_obj_file_path = sample_one_shape_from_(ALL_NORMAL_SHAPES, choose_from = [anomaly_shape_id])

    anomaly_output_folder = get_scene_folder_path(f'{OUT_DIR}/', anomaly_shape_id)

    obj_files = [anomaly_obj_file_path]*3+[normal_obj_file_path]*4
    render_scene_multi(obj_files = obj_files, 
                        out_folder = anomaly_output_folder, 
                        fg_pbr_path = random_sample_one_texture_map_(ALL_MATERIALS_PATH, choose_from = good_ids, empty_prob = 0.1),
                        bg_pbr_path = random_sample_one_texture_map_(ALL_MATERIALS_PATH, empty_prob = 0.02), 
                        env_hdr_path = random_sample_one_texture_map_(ALL_ENVS_PATH),
                        num_views = 8)




# def SceneRenderer(output_path, num_scene = 10, min_objs = 3, max_objs = 10, num_views = 20):
    
#     anomaly_path = '/home/s2514643/MOAD-data-generation-latest/abc_anomaly/'
#     normal_path = '/home/s2514643/MOAD-data-generation-latest/abc_dataset/'

#     model_paths = glob.glob(anomaly_path+'/*')

#     for idx in range(5000):
            
#         path = np.random.choice(model_paths)
        
#         print(f"#### Starting iter {idx}")
#         start_time = time.time()
        
#         model_name = os.path.basename(path)

#         # progress_list = [i[:-1] for i in read_from_log('abc_progress.log')]

#         # if model_name in progress_list:
#         #     print ('scene already rendered')
#         #     continue
        
#         #model_name =  '00098704_56d43bd2e4b0a5e7b6f70056_trimesh_000'
#         path = anomaly_path + '/' + model_name
#         anomaly_obj_files = glob.glob(path+'/*.obj')

#         if len(anomaly_obj_files) == 0:
#             print ('no anomaly')
#             continue
        
#         write_to_log("abc_progress.log", model_name)

#         normal_obj_file = normal_path + '/' + model_name + '.obj'

#         obj_files = [normal_obj_file] + anomaly_obj_files

#         temp_file_name = tempfile.mktemp()+'.npy'

#         new_anomaly_paths, new_normal_path = anomaly_obj_files, normal_obj_file

#         random_angle = [10 + np.random.rand()*10 for _ in range(3)]
        
#         out_list = {}
#         for obj_path in obj_files:
#             rots = simulate_to_get_stable_pose(normal_obj_file, 5, random_angle)
#             matrix = np.eye(4,4)
#             matrix[:3,:3] = rots.T
#             out_list[os.path.basename(obj_path)] = matrix
        
#         np.save(temp_file_name, out_list)

#         for iter_i in range(num_scene):
            
#             output_render_path = get_scene_folder_path(output_path, model_name)

#             if int(output_render_path.split('-')[-1])>=8:
#                 continue

#             n_objs = np.random.choice(np.arange(min_objs, max_objs+1), \
#               )
#             n_anomalies = min(np.random.choice(np.arange(1, max(int(n_objs/2), 2))), len(new_anomaly_paths))
#             random_anomalies = [str(i) for i in np.random.choice(new_anomaly_paths, n_anomalies, replace = False)]
#             normal_list = [new_normal_path]*(n_objs-n_anomalies)  

#             input_paths = random_anomalies + normal_list
#             labels = [0]*len(random_anomalies) + [1]*len(normal_list)
#             try:
#                 ret_code = func_render_scene(input_paths, \
#                         dataset_name = "ABC", \
#                         output_path = output_render_path, \
#                         num_views = num_views, \
#                         pose_file_name = temp_file_name), \
#                         #bg_pbr_path , fg_pbr_path, env_hdr_path)


#                 create_gif_with_labels(output_render_path)

#             except Exception as e:
#                 print(f'err:  Exception: {e}')
            
#         os.remove(temp_file_name)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"#### iter {idx} took {elapsed_time:.2f} seconds\n")
        
        
# if __name__ == "__main__":

#     output_path = '/home/s2514643/abc_test_run_final_v2/'#'/home/s2514643/MOAD-data-generation-latest/output_data/rendered_final_abc_rerun_v2/'

#     min_objs, max_objs = 1, 1
#     num_views = 3
#     num_scene = 4

#     SceneRenderer(output_path, num_scene = num_scene, min_objs = min_objs, max_objs = max_objs, num_views = num_views)

# #obj_paths = ['/home/s2514643/MOAD-data-generation-latest/MOAD-v1/mug_083_ano_v1_reposed.blend','/home/s2514643/MOAD-data-generation-latest/MOAD-v1/mug_083_ano_v1_reposed.blend', '/home/s2514643/MOAD-data-generation-latest/MOAD-v1/mug_083_ano_v2_reposed.blend']

