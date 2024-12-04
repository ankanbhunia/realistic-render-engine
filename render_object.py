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

    render_scene_single(obj_file = '/path/to/obj_file_path', 
                        out_folder = '/path/to/output_folder', 
                        fg_pbr_path = '/path/to/fg_pbr_texture', # Put '' for default object texture.  
                        bg_pbr_path = '/path/to/bg_pbr_texture', # Put '' for white background.  
                        env_hdr_path = '/path/to/hdr_hdr_texture', 
                        num_views = 8)