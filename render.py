import subprocess, shutil, glob, time

from render_utils import *
from anomaly_generators.utils import convert_and_save_as_joined
from anomaly_generators.utils import *


## min_multiplier -> {1,2} high means more space between objects
## radius_constant -> [0.02, 0.1] higher means higher camera distance
## if number of objects in a dataset is higher then above values need to be higher 


def func_render_scene(obj_paths, dataset_name, output_path, min_multiplier=1, radius_constant=0.04, num_views = 20):

    subprocess.call([os.path.abspath("renderings/data_generation/blender-2.93.18-linux-x64/blender"),
    "--background", os.path.abspath("renderings/data_generation/common/empty_scene.blend"), 
    "--python", os.path.abspath("renderings/data_generation/rendering/generate.py"), "--", "--dataset_type", 
    dataset_name, "--output_path", output_path, "--min_multiplier", str(min_multiplier), "--radius_constant", str(radius_constant),
     "--obj_paths_input"]+obj_paths+["--N_FRAMES", str(num_views)], \
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def SceneRenderer(input_path, output_path, num_scene = 10, min_objs = 3, max_objs = 10, num_views = 20):

    model_paths = glob.glob(input_path+'/*')

    for idx in range(100000):
        
        path = np.random.choice(model_paths)

        print(f"#### Starting iter {idx}")
        start_time = time.time()
        
        model_name = os.path.basename(path)

        blend_files = glob.glob(path+'/*.blend')
        normal_blend_file = [i for i in blend_files if "-" not in os.path.basename(i)][0]
        anomaly_blend_files = list(set(blend_files) - {normal_blend_file})

        for iter_i in range(num_scene):

            output_render_path = get_scene_folder_path(output_path, model_name)

            if int(output_render_path.split('-')[-1])>=8:
                continue

            n_objs = np.random.choice(np.arange(min_objs, max_objs+1), p = [0.5, 0.3, 0.2])
            n_anomalies = min(np.random.choice(np.arange(1, int(n_objs/2)+1)), len(anomaly_blend_files))
            random_anomalies = [str(i) for i in np.random.choice(anomaly_blend_files, n_anomalies, replace = False)]
            normal_list = [normal_blend_file]*(n_objs-n_anomalies)

            input_paths = random_anomalies + normal_list
            labels = [0]*len(random_anomalies) + [1]*len(normal_list)
            try:
                ret_code = func_render_scene(input_paths, \
                        dataset_name = "toys", \
                        output_path = output_render_path, \
                        num_views = num_views)

                create_gif_with_labels(output_render_path, input_paths, labels)
            except Exception as e:
                print(f'err:  Exception: {e}')
            

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"#### iter {idx} took {elapsed_time:.2f} seconds\n")
        
        
if __name__ == "__main__":

    path = "/home/s2514643/MOAD-data-generation-latest/output_data/processed_blend_files_final/"
    output_path = '/home/s2514643/MOAD-data-generation-latest/output_data/rendered_final/'

    min_objs, max_objs = 3, 5
    num_views = 20
    num_scene = 1

    SceneRenderer(path, output_path, num_scene = num_scene, min_objs = min_objs, max_objs = max_objs, num_views = num_views)

    #paths = ["/home/s2514643/MOAD-data-generation-latest/output_data/processed_blend_files/apple_050/apple_050-crack-002.blend",
    #"/home/s2514643/MOAD-data-generation-latest/output_data/processed_blend_files/apple_050/apple_050.blend",
    #"/home/s2514643/MOAD-data-generation-latest/output_data/processed_blend_files/apple_050/apple_050.blend",
    #]
    #out_path = "/home/s2514643/MOAD-data-generation-latest/MOAD-v1/out_test/"
    #func_render_scene(paths, 'toys',  get_scene_folder_path(out_path, 'model'), num_views = 10)

    # #convert_and_save_as_joined("/disk/nfs/gazinasvolume2/s2514643/MOAD-v1/models/shoe-defr-001.blend")    
    # input_paths = ["/disk/nfs/gazinasvolume1/datasets/toys4k_blend_files/panda/panda_011/panda_011.blend"]*5+\
    #     ["/disk/nfs/gazinasvolume2/s2514643/MOAD-v1/models/panda-mats-002.blend"]*1
    # labels = [1,1,1,1,1,0]

    # output_render_path = '/disk/nfs/gazinasvolume2/s2514643/MOAD-v1/output'
    
    # if os.path.isdir(output_render_path): 
    #     shutil.rmtree(output_render_path)
    # os.makedirs(output_render_path)

    # render_scene(input_paths, \
    #     dataset_name = "toys", \
    #     output_path = output_render_path)

    # create_gif_with_labels( f'{output_render_path}/out.mp4', f'{output_render_path}', input_paths, labels)




re_pose_blend_file('/home/s2514643/MOAD-data-generation-latest/MOAD-v1/mug_083_ano.blend', \
    '/home/s2514643/MOAD-data-generation-latest/MOAD-v1/mug_083_ano_reposed.blend', plane_height = 0.05, rot_degree = [0,0,0])

obj_paths = ['/disk/scratch/s2514643/toys4k_blend_files/mug/mug_083/mug_083.blend', \
    '/disk/scratch/s2514643/toys4k_blend_files/mug/mug_083/mug_083.blend',\
        '/home/s2514643/MOAD-data-generation-latest/MOAD-v1/mug_083_ano_reposed.blend']
func_render_scene(obj_paths, 'toys', \
    '/home/s2514643/MOAD-data-generation-latest/MOAD-v1/out_test_v2/',\
         min_multiplier=1, radius_constant=0.04, num_views = 20)