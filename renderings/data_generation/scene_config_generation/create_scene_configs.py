import os
import numpy as np
import json

from scipy.spatial.distance import pdist
import time

# CONSTANTS FOR DATA GENRATION
DATASET = "shapenet"
ASSET_JSON_FILE = "../common/jsons/scene_assets_new.json"
OBJ_JSON_FILE = "../common/jsons/{}_dict.json".format(DATASET)
OBJ_SPLIT_JSON_FILE = "../common/jsons/shapenet_cate_split.json"
OBJ_POSE_DIR = "../common/{}_poses_not_canonical".format(DATASET)
OUTPUT_DIR = "../common/{}_scene_configs_test".format(DATASET)
N_FRAMES = 20
N_START = 0
N_SCENES = 10
OBJ_X_MIN = -0.2
OBJ_X_MAX = 0.2
OBJ_Y_MIN = -0.2
OBJ_Y_MAX = 0.2
# OBJ_COUNT_MIN = 2
# OBJ_COUNT_MAX = 7
OBJ_COUNT_MIN = 2
OBJ_COUNT_MAX = 2
OBJ_SCALE_MIN = 0.35
OBJ_SCALE_MAX = 0.45
CAM_RADIUS_MIN = 0.6  #0.45
CAM_RADIUS_MAX = 0.8
CAM_HEIGHT_MIN = 0.35 #0.35
CAM_HEIGHT_MAX = 0.65 #0.65
CAM_ANGLES = "RANDOM UNIFORM" # OR "FIXED UNIFORM"
JITTERING = 0.01

# CAM_ANGLES = "FIXED UNIFORM" # OR "FIXED UNIFORM"
# CAM_ANGLES = "INTERPOLATE"

BRIGHT_MIN = 0.4
BRIGHT_MAX = 0.8
#### Nearest distance 2 objects can be
MARGIN = 0.4
RENDERING_MODE = 'TRAIN_SHOT'

def load_obj_list(mode):
    with open(OBJ_SPLIT_JSON_FILE, "r") as f:
        obj_dict = json.load(f)
    
    if mode == 'BASE':
        keys = ['base_1', 'others']
    elif mode == 'TRAIN_SHOT':
        # keys = ['base_1', 'base_2', 'train_shot']
        ##### Level 1
        keys = ['base_1', 'train_shot']

    elif mode == 'TEST_SHOT':
        # keys = ['base_1', 'base_2', 'others', 'test_shot']
        ##### Level 1
        keys = ['test_shot']

    
    objects = []
    objects2 = []
    for k in keys:
        d = obj_dict[k]
        for category, instances in d.items():
            if k == 'train_shot' or k == 'test_shot':
                object_paths2 = [os.path.join(category, instance, 'models', 'model_normalized.obj') for instance in instances]
                objects2.extend(object_paths2)
            else:
                object_paths = [os.path.join(category, instance, 'models', 'model_normalized.obj') for instance in instances]
                objects.extend(object_paths)
    
    return objects, objects2

def pick_object_location(n, min_x, max_x, min_y, max_y, margin, thres=0.3):
    """
        Sample (x,y) positions so no two samples are closer than margin

        positional
        n -- how many points to sample
        min_x -- minimum x coordinate
        max_x -- maximum y coordinate
        min_y -- minimum x coordinate
        max_y -- maximum y coordinate
    """
        
    count = 0
    locations = []
    start = time.time()
    while count < n and time.time()-start <= thres:

        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        
        locations.append([x,y])

        if count >= 1:
            arr = np.array(locations)
        
            d = pdist(arr)
            ## if added point is too close to any other point try again
            if np.any(d<=margin):
                locations.pop(-1)
                continue
        
        count+=1
    if count < n: ### Timeout, restart scene
        return -1, -1

    means = np.mean(locations, axis=0)

    return locations, means

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

def pick_object_pose(obj):
    
    category, obj_id = get_id_info(obj, DATASET)
    
    pose_json_path = os.path.join(
            OBJ_POSE_DIR, 
            category, 
            obj_id, 
            "pose_list.json"
        )
    try:
        with open(pose_json_path, "r") as f:
            pose_list = json.load(f)

        pose = np.random.choice(pose_list)
        # pose = pose_list[0]

        pose = pose["rotation_matrix"]
    except:
        pose = np.eye(3)
    
    theta = np.random.uniform(-2*np.pi, 2*np.pi)
    random_z_rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [            0,              0, 1]
        ])
    
    pose = random_z_rotation @ np.array(pose)
    ##### Only rotate around z
    # pose = random_z_rotation

    pose = pose.tolist()
    
    return pose
def main():
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(ASSET_JSON_FILE, "r") as f:
        assets_dict = json.load(f)

    hdr_files = assets_dict["hdr_files"]
    pbr_files = assets_dict["pbr_files"]
    
    obj_list, obj_list2 = load_obj_list(RENDERING_MODE)

    n = N_START
    start_time = time.time()
    
    while n < N_SCENES + N_START and n >= N_START:

        # Choose floor appearance
        floor_pbr = np.random.choice(pbr_files)

        # Choose background appearance
        bg_hdr = np.random.choice(hdr_files) 

        object2 = [] 

        # Choose objects
        if len(obj_list2) > 0:
            if RENDERING_MODE == 'TRAIN_SHOT':
                obj_count = np.random.randint(OBJ_COUNT_MIN-1, OBJ_COUNT_MAX)
                object2 = list([np.random.choice(obj_list2)])

            else:
                obj_count = np.random.randint(0, OBJ_COUNT_MAX)
                object2 = list(np.random.choice(obj_list2, OBJ_COUNT_MAX-obj_count, replace=False))

        else:
            obj_count = np.random.randint(OBJ_COUNT_MIN, OBJ_COUNT_MAX+1)
        objects = list(np.random.choice(obj_list, obj_count, replace=False))

        if len(object2) > 0:
            objects.extend(object2)
            obj_count += len(object2)
            print('shot')


        # Choose object locations (T)
        object_locations, means = pick_object_location(
            obj_count,
            OBJ_X_MIN,
            OBJ_X_MAX,
            OBJ_Y_MIN,
            OBJ_Y_MAX,
            MARGIN)

        if object_locations == -1 and means == -1:
            print("Regenerating scene: ", n)
            continue

        # Choose object poses (R)
        object_poses = [pick_object_pose(obj) for obj in objects]

        # Choose object scales
        object_scales = np.random.uniform(OBJ_SCALE_MIN, OBJ_SCALE_MAX, obj_count)

        # Choose camera poses (T) 
        if CAM_ANGLES == "FIXED UNIFORM":
            angles = np.linspace(0.0, 2*np.pi, N_FRAMES)
        if CAM_ANGLES == "RANDOM UNIFORM":
            angles = np.random.uniform(low=0.0, high=2*np.pi, size=N_FRAMES)
        radii = np.random.uniform(low=CAM_RADIUS_MIN, high=CAM_RADIUS_MAX, size=N_FRAMES)
        x_cam_co = radii*np.cos(angles)
        y_cam_co = radii*np.sin(angles)
        z_cam_co = np.random.uniform(CAM_HEIGHT_MIN, CAM_HEIGHT_MAX, N_FRAMES)
        camera_positions = np.stack([x_cam_co, y_cam_co, z_cam_co]).T
    
        # Choose camera track-to point
        track_to_points = np.zeros((N_FRAMES, 3))

        track_to_points[:,0:2] = track_to_points[:,0:2] + means.reshape(1,-1) + \
            np.random.uniform(-JITTERING, JITTERING, N_FRAMES*2).reshape(N_FRAMES,-1)

        # track_to_points[:,0:2] = track_to_points[:,0:2] + means.reshape(1,-1)

        
        # Choose environment strenght (brighness)
        strength = np.random.uniform(0.2, 0.8)

        # Populating scene dictionary
        object_output_list = []
        for i in range(obj_count):
           obj_dict = {
                "obj_subpath":objects[i],
                "obj_location":object_locations[i],
                "obj_pose":object_poses[i],
                "obj_scale":object_scales[i],
            }
           object_output_list.append(obj_dict)
        
        camera_dict = {
            "track_to_point":track_to_points.tolist(),
            "positions":camera_positions.tolist()
        }

        appearance_assets_dict = {
            "floor_pbr":floor_pbr,
            "background_hdr":bg_hdr
        }
        
        scene_dict = {
            "objects":object_output_list,
            "camera":camera_dict,
            "appearance_assets":appearance_assets_dict,
            "environment_strength":strength
        }
        
        out_str = json.dumps(scene_dict, indent=True)
        with open(os.path.join(OUTPUT_DIR, "{:05d}.json".format(n)), "w") as f:
            f.write(out_str)

        if n%1000 == 0:
            print("Generated %s scenes"%(n))

        n += 1
    end_time = time.time()
    print("Finished generating %s scenes in %ss"%(N_SCENES, (end_time-start_time)))

if __name__ == "__main__":
    main()
