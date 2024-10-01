import anomaly_generators.operations as agp 
import random, glob, os, shutil, time, argparse
from anomaly_generators.utils import *

#re_pose_blend_file('/disk/nfs/gazinasvolume2/s2514643/MOAD-v1/test_temp/mug_068/mug_068-bump-001.blend', '/disk/nfs/gazinasvolume2/s2514643/MOAD-v1/im.blend', plane_height = 0.5)

def run_function(func, blend_file_path, output_folder, *args, **kwargs):
    try:
        ret_status = func(blend_file_path, output_folder, *args, **kwargs)
        if not ret_status:
            print(f'wrn: no anomaly created while executing with arguments {args}')
        else: 
            shutil.copy(blend_file_path, output_folder)
    except Exception as e:
        print(f'err: error while executing with arguments {args}. Exception: {e}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Anomaly-Generation-script")
    parser.add_argument("--anomaly_type", type=str) # fracture, 
    parser.add_argument("--category", type=str, default = 'all')
    parser.add_argument("--total_iterations", type=int, default = 500)
    parser.add_argument("--dataset_path", type=str, default = "/disk/scratch/s2514643/toys4k_blend_files/")
    parser.add_argument("--out_path", type=str, default = "./out_folder")

    args = parser.parse_args()

    if args.category == 'all': args.category = "*"

    for idx in range(args.total_iterations):

        print(f"#### Starting iter {idx}")
        start_time = time.time()

        blend_file_path = random.choice(glob.glob(f'{args.dataset_path}/{args.category}/*/*'))
        model_name = blend_file_path.split('/')[-2]
        output_folder = f"{args.out_path}/{model_name}/"

        if args.anomaly_type == "fracture":
            run_function(agp.FractureAnomalyGenerator, blend_file_path, output_folder, model_name, \
                min_region = 0.02, max_region = 1.0, modes = ['frac', 'crack', 'disc'], num_iters = 500)

        if args.anomaly_type == "bump":
            run_function(agp.BumpAnomalyGenerator, blend_file_path, output_folder, model_name, \
                mode = 'bump')

        if args.anomaly_type == "deform":
            run_function(agp.HookDeformAnomalyGenerator, blend_file_path, output_folder, model_name, \
                mode = 'defr')

        if args.anomaly_type == "bend/twist":
            run_function(agp.SimpleDeformAnoamlyGenerator, blend_file_path, output_folder, model_name, \
                modes = ['bend', 'twist'], min_angle = 25, max_angle = 30)

        if args.anomaly_type == "missing":
            run_function(agp.MissingAnomalyGenerator, blend_file_path, output_folder, model_name, \
                mode = 'miss', min_region = 0.01, max_region=0.5, num_iters = 500)

        if args.anomaly_type == "3dTransform":
            run_function(agp.Transform3DAnomalyGenerator, blend_file_path, output_folder, model_name, \
                min_region = 0.01, max_region = 0.5, num_iters = 500,  \
                min_trans = 0.1, max_trans = 0.4, min_angle =  0.3, max_angle = 0.4, \
                min_scale = 0.7, max_scale = 0.8, max_try = 10, \
                modes = ['tran', 'rota', 'scal'])

        if args.anomaly_type == "material":
            source_material_path = random.choice(glob.glob(f'{args.dataset_path}/*/*/*'))
            run_function(agp.Transform3DAnomalyGenerator, blend_file_path, output_folder, model_name, \
                source_material_path = source_material_path, \
                modes = ['mats'])

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"#### iter {idx} took {elapsed_time:.2f} seconds\n")


