import os
import time
import argparse
import json
import numpy as np

os.chdir('LSME/data_generation/rendering')

def get_id_info(path, dataset_type):
    if dataset_type == "modelnet":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-1][:-4]
        return category, obj_id

    if dataset_type == "shapenet":
        category = path.split("/")[-4]
        obj_id = path.split("/")[-3]
        return category, obj_id

    if dataset_type == "toys":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-2]
        return category, obj_id

def main():
    
    # set paths
    blender_script_path = os.path.abspath("generate.py")
    blendfile_path = os.path.abspath("../common/empty_scene.blend")
    
    # load arguments
    parser = argparse.ArgumentParser(description="Range of Objects")
    parser.add_argument("--start", type=int, help="start point")
    parser.add_argument("--end", type=int, help="end point")
    parser.add_argument("--dataset_path", type=str, help="dataset input path")
    parser.add_argument("--output_path", type=str, help="render output path")
    parser.add_argument("--blender_path", type=str, help="path to blender")
    parser.add_argument("--config_path", type=str, help="path to config files")
    parser.add_argument(
        "--dataset_type", type=str, help="either <modelnet>, <shapenet> or <toys>"
    )
    
    args = parser.parse_args()

    print("Start index: {}".format(args.start))
    print("End index: {}".format(args.end))
    print("Input path: {}".format(args.dataset_path))
    print("Output path: {}".format(args.output_path))
    print("Dataset type: {}".format(args.dataset_type))
    
    # config_paths = [os.path.join(args.config_path, x) for x in sorted(os.listdir(args.config_path))]
    
    # config_paths = config_paths[args.start : min(len(config_paths), args.end)]
    

    # # missing_files = np.load('/home/ant/data/odme/ABC_rendering_output/error.npz', allow_pickle=True)['errors']
    # # config_paths = [os.path.join(args.config_path, x.split('/')[-1]+'.json') for x in missing_files]
    # # import pdb; pdb.set_trace()

     
    # print("will render {} items".format(len(config_paths)))
    i = 0
    global_time = time.time()
    
    # render each object
    for idx in range(10):

        start_time = time.time()

        cmd = (
            f"{args.blender_path} -noaudio --background {blendfile_path} --python {blender_script_path} -- "
            f"--dataset_type {args.dataset_type} "
            f"--output_path {args.output_path} "
            f"--dataset_path {args.dataset_path} "
            f"--cat_id {args.cat_id} "
            f"--obj_count {args.obj_count} "
            f"--N_FRAMES {args.N_FRAMES} "
            f"--OBJ_SCALE_MIN {args.OBJ_SCALE_MIN} "
            f"--OBJ_SCALE_MAX {args.OBJ_SCALE_MAX} "
            f"--CAM_RADIUS_MIN {args.CAM_RADIUS_MIN} "
            f"--CAM_RADIUS_MAX {args.CAM_RADIUS_MAX} "
            f"--CAM_HEIGHT_MIN {args.CAM_HEIGHT_MIN} "
            f"--CAM_HEIGHT_MAX {args.CAM_HEIGHT_MAX} "
            f"1>/tmp/out.txt"
        )
        print(cmd)
        os.system(cmd)

        # print(
        #     "--- {:.2f} seconds for obj {} [{}/{}] ---".format(
        #         time.time() - start_time, cpath, args.start + i, args.end
        #     )
        # )

        if i % 50 == 0:
            print(
                "Total time since start is {:.2f} minutes".format(
                    (time.time() - global_time) / 60
                )
            )
        i += 1

    total_time = time.time() - global_time
    print("Total time {:.2f}".format(total_time / 60))


if __name__ == "__main__":
    main()
