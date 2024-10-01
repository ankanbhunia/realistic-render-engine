import subprocess, shutil, glob, time, cv2


from render_utils import *
from anomaly_generators.utils import convert_and_save_as_joined
from anomaly_generators.utils import *
from renderings.utils import generate_localization_maps
from concurrent.futures import ProcessPoolExecutor


def render_scene_single(obj_file, out_folder, fg_pbr_path, bg_pbr_path, env_hdr_path, num_views = 3):

    def func_render_scene(obj_paths, dataset_name, output_path, pose_file_name, bg_pbr_path, fg_pbr_path, env_hdr_path, num_views = 20):

        subprocess.call([os.path.abspath("renderings/data_generation/blender-2.93.18-linux-x64/blender"),
        "--background", os.path.abspath("renderings/data_generation/common/empty_scene.blend"), 
        "--python", os.path.abspath("renderings/data_generation/rendering/generate_single_abc.py"), "--", "--dataset_type", 
        dataset_name, "--output_path", output_path, "--pose_file_name", pose_file_name,
        "--bg_pbr_path", str(bg_pbr_path), "--fg_pbr_path", str(fg_pbr_path), "--env_hdr_path", str(env_hdr_path),
        "--obj_paths_input",]+obj_paths+["--N_FRAMES", str(num_views)], \
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    temp_file_name = tempfile.mktemp()+'.npy'
    rots = simulate_to_get_stable_pose(obj_file, 5, [10 + np.random.rand()*10 for _ in range(3)], num_frames = 1000)
    matrix = np.eye(4,4)
    matrix[:3,:3] = rots.T
    np.save(temp_file_name, {os.path.basename(obj_file):matrix})
    
    ret_code = func_render_scene([obj_file], \
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

    OUT_DIR = '/disk/nfs/gazinasvolume2/s2514643/look3d-o3-rendering/data/abc_rendering_reuslts/'

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f'{OUT_DIR}/normal/', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/anomaly/', exist_ok=True)

    good_ids = ['-'.join(i[:-5].split('-')[1:]) for i in open('/disk/nfs/gazinasvolume2/s2514643/look3d-o3-rendering/data/good_ids.txt','r').readlines()]
        #### generete anomaly scene 
        
    def task_(idx):

        try:

            anomaly_obj_file_path = sample_one_shape_from_(ALL_ANOMALY_SHAPES)

            anomaly_shape_id = anomaly_obj_file_path.split('/')[-2]
            anomaly_output_folder = get_scene_folder_path(f'{OUT_DIR}/anomaly/', os.path.basename(anomaly_obj_file_path)[:-4])

            render_scene_single(obj_file = anomaly_obj_file_path, 
                                out_folder = anomaly_output_folder, 
                                fg_pbr_path = random_sample_one_texture_map_(ALL_MATERIALS_PATH, choose_from = good_ids, empty_prob = 0.1),
                                bg_pbr_path = random_sample_one_texture_map_(ALL_MATERIALS_PATH, empty_prob = 0.02), 
                                env_hdr_path = random_sample_one_texture_map_(ALL_ENVS_PATH),
                                num_views = 8)


            #### generete normal scene 

            if len([i for i in os.listdir(f'{OUT_DIR}/normal/') if anomaly_shape_id in i]) == 0:
                normal_obj_file_path = sample_one_shape_from_(ALL_NORMAL_SHAPES, choose_from = [anomaly_shape_id])
            else:
                normal_obj_file_path = sample_one_shape_from_(ALL_NORMAL_SHAPES)
            normal_output_folder = get_scene_folder_path(f'{OUT_DIR}/normal/', os.path.basename(normal_obj_file_path)[:-4])

            render_scene_single(obj_file = normal_obj_file_path, 
                                out_folder = normal_output_folder, 
                                fg_pbr_path = random_sample_one_texture_map_(ALL_MATERIALS_PATH, choose_from = good_ids, empty_prob = 0.1),
                                bg_pbr_path = random_sample_one_texture_map_(ALL_MATERIALS_PATH, empty_prob = 0.02), 
                                env_hdr_path = random_sample_one_texture_map_(ALL_ENVS_PATH),
                                num_views = 8)

            #### generete location maps for the anomaly

            generate_localization_maps(anomaly_path = anomaly_output_folder, \
                ref_normal_path = sample_one_shape_from_(glob.glob(f'{OUT_DIR}/normal/*'), choose_from = [anomaly_shape_id]))

        except:

            print ('Error Occurred!')
            os.system(f'rm -rf {normal_output_folder}')
            os.system(f'rm -rf {anomaly_output_folder}')


    # while True:


    #     task_()

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(task_, np.arange(100000)))

# all_files = glob.glob('/disk/nfs/gazinasvolume2/s2514643/MatSynth/*/*/*/roughness.png')
# choose_from = ['-'.join(i[:-5].split('-')[1:]) for i in open('/disk/nfs/gazinasvolume2/s2514643/look3d-o3-rendering/data/good_ids.txt','r').readlines()]
# all_files_ = [i for i in all_files if i.split('/')[-2] in choose_from]
# r = []
# for i in tqdm.tqdm(all_files_):
#     r.append((cv2.imread(i)/255).mean())
    