import os
import numpy as np
import json
import colorsys

from scipy.spatial.distance import pdist
import time

################# HARDCODING BELOW

# CONSTANTS FOR DATA GENRATION
DATASET = "ABC" # toys, shapenet or modelnet
ASSET_JSON_FILE = "../common/jsons/scene_assets_new.json"
OBJ_JSON_FILE = "../common/jsons/{}_glb_125K.json".format(DATASET)
OBJ_POSE_DIR = "/data/DevLearning/ABC_poses_GLB_125K"
OUTPUT_DIR = "../common/{}_scene_configs_2023-02-14".format(DATASET)

'''
R_MIN = 1.75
R_MAX = 4.0

OBJ_SHIFT_MIN = -0.5
OBJ_SHIFT_MAX = 0.5

CAM_TRACK_SHIFT_MIN = -0.5
CAM_TRACK_SHIFT_MAX = 0.5

AZIM_MIN = 0
AZIM_MAX = 360

ELEV_MIN = 10
ELEV_MAX = 80

OBJ_SCALE_MIN = 1.0
OBJ_SCALE_MAX = 1.0

PLANE_POS_MIN = -4.0
PLANE_POS_MAX = 4.0
'''

N_FRAMES = 20
N_SCENES = 5
# N_SCENES = 100


# R_MIN = 1.75
# R_MAX = 4.0

CAM_RADIUS_MIN = 1.1
CAM_RADIUS_MAX = 1.4
CAM_HEIGHT_MIN = 0.3
CAM_HEIGHT_MAX = 0.5

OBJ_X_MIN = -0.5
OBJ_X_MAX = 0.5
OBJ_Y_MIN = -0.5
OBJ_Y_MAX = 0.5
# OBJ_X_MIN = 0
# OBJ_X_MAX = 0
# OBJ_Y_MIN = 0
# OBJ_Y_MAX = 0

OBJ_COUNT_MIN = 2
OBJ_COUNT_MAX = 2
# OBJ_COUNT_MIN = 3
# OBJ_COUNT_MAX = 7

# CAM_ANGLES = "INTERPOLATE"
CAM_ANGLES = "RANDOM UNIFORM"


# OBJ_SCALE_MIN = 0.2
# OBJ_SCALE_MAX = 0.4

OBJ_SCALE_MIN = 0.3
OBJ_SCALE_MAX = 0.5

# OBJ_SHIFT_MIN = 0.0
# OBJ_SHIFT_MAX = 0.0

# CAM_TRACK_SHIFT_MIN = -0.5
# CAM_TRACK_SHIFT_MAX = -0.5

# AZIM_MIN = 0
# AZIM_MAX = 360

# ELEV_MIN = 10
# ELEV_MAX = 80

# OBJ_SCALE_MIN = 1.0
# OBJ_SCALE_MAX = 1.0

# PLANE_POS_MIN = -4.0
# PLANE_POS_MAX = 4.0

BRIGHT_MIN = 0.4
BRIGHT_MAX = 0.9
#### Nearest distance 2 objects can be
MARGIN = 0.4

def load_obj_list():
    with open(OBJ_JSON_FILE, "r") as f:
        obj_dict = json.load(f)

    objects = []
    for category, object_paths in obj_dict.items():
        object_paths = [
            os.path.join(category, x) for x in object_paths
        ]
        objects.extend(object_paths)

    return objects

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

    if dataset_type == "ABC":
        obj_id = path.split("/")[-1]
        category = ''

        return category, obj_id

def pick_object_pose(obj):

    category, obj_id = get_id_info(obj, DATASET)
    # print(category, obj_id)

    pose_json_path = os.path.join(
            OBJ_POSE_DIR,
            category,
            obj_id,
            "pose_list.json"
        )
    with open(pose_json_path, "r") as f:
        pose_list = json.load(f)

    pose = np.random.choice(pose_list)
    pose = pose["rotation_matrix"]

    theta = np.random.uniform(-2*np.pi, 2*np.pi)
    random_z_rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [            0,              0, 1]
        ])

    pose = random_z_rotation @ np.array(pose)
    pose = pose.tolist()

    #pose = np.eye(3)
    #pose = pose.tolist()

    return pose

def pick_color():

    out_dct = {}
    types = ["single", "single", "random-multi", "random-mono"]
    if np.random.rand() < 0.2:
        clever = True
    else:
        clever = False

    color_type = np.random.choice(types)

    out_dct["type"] = color_type

    if color_type == "single":

        if np.random.choice([0,1]):
            h,s,l = np.random.rand(), 0.2 + 0.4*np.random.rand(), np.random.rand()
            r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]
            out_dct["color"] = [r,g,b]
        else:
            val = np.random.rand()
            out_dct["color"] = [val, val, val]

    if color_type == "random-multi":
        n_colors = np.random.randint(2,5)

        color = []
        for i in range(n_colors):
            h,s,l = np.random.rand(), 0.6 + 0.4*np.random.rand(), 0.3 + np.random.rand()/4.0
            r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]
            color.append([r,g,b])

        out_dct["color"] = color
        out_dct["randomness"] = np.random.uniform(0,5)
        out_dct["scale"] = np.random.uniform(0.5,1.5)

    if color_type == "random-mono":
        n_colors = np.random.randint(2,5)

        color = []
        h = np.random.rand()
        for i in range(n_colors):
            s,l = 0.2 + 0.4*np.random.rand(), np.random.rand()
            r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]
            color.append([r,g,b])

        out_dct["color"] = color
        out_dct["randomness"] = np.random.uniform(0,1)
        out_dct["scale"] = np.random.uniform(2,5)


    out_dct["specular"] = np.random.uniform(0.6,0.9)
    out_dct["roughness"] = np.random.uniform(0.1,0.3)

    out_dct["clever"] = clever
        

    return out_dct

def sample_pair():

    plane_loc = np.random.uniform(
            low=PLANE_POS_MIN,
            high=PLANE_POS_MAX,
            size=3)
    plane_loc[-1] = 0

    azim1 = np.random.uniform(
            low=AZIM_MIN,high=AZIM_MAX)
    elev1 = np.random.uniform(
            low=ELEV_MIN,high=ELEV_MAX)

    azim2 = azim1 + np.random.uniform(low=-90, high=90)
    elev2 = np.random.uniform(
            low=ELEV_MIN,high=ELEV_MAX)

    azim1 = np.radians(azim1)
    azim2 = np.radians(azim2)

    elev1 = np.radians(elev1)
    elev2 = np.radians(elev2)

    r1 = np.random.uniform(low=R_MIN, high=R_MAX)
    r2 = np.random.uniform(low=R_MIN, high=R_MAX)

    x1 = r1 * np.cos(azim1) * np.sin(elev1)
    y1 = r1 * np.sin(azim1) * np.sin(elev1)
    z1 = r1 * np.cos(elev1)

    x2 = r2 * np.cos(azim2) * np.sin(elev2)
    y2 = r2 * np.sin(azim2) * np.sin(elev2)
    z2 = r2 * np.cos(elev2)

    cam_pos1 = plane_loc + np.array([x1,y1,z1])
    cam_pos2 = plane_loc + np.array([x2,y2,z2])

    track_to_loc1 = plane_loc + np.random.uniform(
        low=CAM_TRACK_SHIFT_MIN,
        high=CAM_TRACK_SHIFT_MAX,
        size=3)

    track_to_loc1[-1] = np.random.uniform(low=0, high=1)

    track_to_loc2 = plane_loc + np.random.uniform(
        low=CAM_TRACK_SHIFT_MIN,
        high=CAM_TRACK_SHIFT_MAX,
        size=3)

    track_to_loc2[-1] = np.random.uniform(low=0, high=1)

    obj_shift = np.random.uniform(
            low=OBJ_SHIFT_MIN,
            high=OBJ_SHIFT_MAX,
            size=3)
    obj_shift[-1] = 0

    obj_pos_1 = plane_loc + obj_shift
    obj_pos_2 = plane_loc + obj_shift

    to_return = (cam_pos1, cam_pos2, obj_pos_1, obj_pos_2, track_to_loc1, track_to_loc2)

    return to_return

def main():

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("output:", OUTPUT_DIR)

    with open(ASSET_JSON_FILE, "r") as f:
        assets_dict = json.load(f)

    hdr_files = assets_dict["hdr_files"]
    pbr_files = assets_dict["pbr_files"]

    # pbrs_to_keep = ['Tiles027', 'Tiles071', 'WoodFloor05']
    # hdrs_to_keep = ['cayley_interior_4k.hdr', 'anniversary_lounge_4k.hdr']
    # hdr_files = [x for x in hdr_files if x in hdrs_to_keep]
    # pbr_files = [x for x in pbr_files if x in pbrs_to_keep]
    '''
    pbrs_to_keep = ['WoodFloor005']
    hdrs_to_keep = ['cayley_interior_4k.hdr']

    hdr_files = [x for x in hdr_files if x in hdrs_to_keep]
    pbr_files = [x for x in pbr_files if x in pbrs_to_keep]
    '''
    ##### Load object list
    with open(OBJ_JSON_FILE, "r") as f:
        obj_dict = json.load(f)

    obj_list = []

    for k, v in obj_dict.items():
        obj_list.extend(sorted(v))
    n = 0
    start_time = time.time()

    ################ Hardcode for same obj, comment out the same thing in the loop
    # obj_count = np.random.randint(OBJ_COUNT_MIN, OBJ_COUNT_MAX+1)

    # objects = np.random.choice(obj_list, obj_count, replace=False)
    ################ Hardcode

    while n < N_SCENES:

        # Choose floor appearance
        floor_pbr = np.random.choice(pbr_files)

        # Choose background appearance
        bg_hdr = np.random.choice(hdr_files)

        # Choose objects
        obj_count = np.random.randint(OBJ_COUNT_MIN, OBJ_COUNT_MAX+1)
        # print(obj_count)
        objects = np.random.choice(obj_list, obj_count, replace=False)



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
        # print(object_scales)


        # Choose camera poses (T) 
        if CAM_ANGLES == "FIXED UNIFORM":
            # angles = np.linspace(0.0, 2*np.pi, N_FRAMES)
            angles = np.array([0.0]*N_FRAMES)
        if CAM_ANGLES == "RANDOM UNIFORM":
            angles = np.random.uniform(low=0.0, high=2*np.pi, size=N_FRAMES)
        if CAM_ANGLES == "FIXED UNIFORM" or CAM_ANGLES == "RANDOM UNIFORM":
            radii = np.random.uniform(low=CAM_RADIUS_MIN, high=CAM_RADIUS_MAX, size=N_FRAMES)
            x_cam_co = radii*np.cos(angles)
            y_cam_co = radii*np.sin(angles)
            z_cam_co = np.random.uniform(CAM_HEIGHT_MIN, CAM_HEIGHT_MAX, N_FRAMES)
            camera_positions = np.stack([x_cam_co, y_cam_co, z_cam_co]).T

        if CAM_ANGLES == "INTERPOLATE":
            alow = np.random.uniform(low=0.0, high=2*np.pi)
            if np.random.random() < 0.5:
                ahigh = alow - (np.random.random()*2*np.pi/5+np.pi/10)
            else:
                ahigh = alow + (np.random.random()*2*np.pi/5+np.pi/10)
            # ahigh = np.random.uniform(low=0.0, high=2*np.pi)
            rlow = np.random.uniform(CAM_RADIUS_MIN, CAM_RADIUS_MAX)
            rhigh = np.random.uniform(CAM_RADIUS_MIN, CAM_RADIUS_MAX)

            x_low = rlow*np.cos(alow)
            y_low = rlow*np.sin(alow)
            z_low = np.random.uniform(CAM_HEIGHT_MIN, CAM_HEIGHT_MAX)

            x_high = rhigh*np.cos(ahigh)
            y_high = rhigh*np.sin(ahigh)
            z_high = np.random.uniform(CAM_HEIGHT_MIN, CAM_HEIGHT_MAX)

            x_cam_co = np.linspace(x_low, x_high, N_FRAMES)
            y_cam_co = np.linspace(y_low, y_high, N_FRAMES)
            z_cam_co = np.linspace(z_low, z_high, N_FRAMES)

            camera_positions = np.stack([x_cam_co, y_cam_co, z_cam_co]).T
        
        # Choose camera track-to point
        track_to_points = np.zeros((N_FRAMES, 3))
        # print(means)
        # track_to_points[:,0:2] = track_to_points[:,0:2] + means.reshape(1,-1) + \
        #     np.random.uniform(-0.03, 0.03, N_FRAMES*2).reshape(N_FRAMES,-1)
        track_to_points[:,0:2] = track_to_points[:,0:2] + means.reshape(1,-1)



        # Choose environment strength (brighness)
        strength = np.random.uniform(BRIGHT_MIN, BRIGHT_MAX)


        color_dicts = [pick_color() for _ in range(obj_count)]


        # Populating scene dictionary
        object_output_list = []
        for i in range(obj_count):
           obj_dict = {
                "obj_subpath":objects[i],
                "obj_location":object_locations[i],
                "obj_pose":object_poses[i],
                "obj_scale":object_scales[i],
                "obj_color":color_dicts[i]
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


#     obj_count = 1

#     for obj in obj_list:

#         categ = ""
#         obj = os.path.join(categ, obj)

#         print(categ, obj)

#         # Choose floor appearance
#         floor_pbr = np.random.choice(pbr_files)

#         # Choose background appearance
#         bg_hdr = np.random.choice(hdr_files)

#         # Choose objects
#         objects = [obj]

#         # Choose object poses (R)
#         object_poses = [pick_object_pose(obj) for obj in objects]

#         # Choose object scales
#         object_scales = np.random.uniform(OBJ_SCALE_MIN, OBJ_SCALE_MAX, 1)

#         track_to_points = []
#         object_locations = []
#         camera_positions = []

#         for i in range(10):
#             out = sample_pair()
#             cam_pos_1, cam_pos_2, obj_pos_1, obj_pos_2, track_to_loc_1, track_to_loc_2 = out

#             camera_positions.append(cam_pos_1)
#             camera_positions.append(cam_pos_2)

#             object_locations.append(obj_pos_1)
#             object_locations.append(obj_pos_2)

#             track_to_points.append(track_to_loc_1)
#             track_to_points.append(track_to_loc_2)

#         track_to_points = np.array(track_to_points)
#         camera_positions = np.array(camera_positions)
#         object_locations = np.array(object_locations).tolist()
#         object_locations = [object_locations]

#         # Choose environment strength (brighness)
#         strength = np.random.uniform(BRIGHT_MIN, BRIGHT_MAX)

#         color_dict = pick_color()

#         # Populating scene dictionary
#         object_output_list = []
#         for i in range(obj_count):
#            obj_dict = {
#                 "obj_subpath":objects[i],
#                 "obj_location":object_locations[i],
#                 "obj_pose":object_poses[i],
#                 "obj_scale":object_scales[i],
#                 "obj_color":color_dict
#             }
#            object_output_list.append(obj_dict)

#         camera_dict = {
#             "track_to_point":track_to_points.tolist(),
#             "positions":camera_positions.tolist()
#         }

#         appearance_assets_dict = {
#             "floor_pbr":floor_pbr,
#             "background_hdr":bg_hdr
#         }

#         scene_dict = {
#             "objects":object_output_list,
#             "camera":camera_dict,
#             "appearance_assets":appearance_assets_dict,
#             "environment_strength":strength
#         }

#         out_str = json.dumps(scene_dict, indent=True)
#         with open(os.path.join(OUTPUT_DIR, "{}.json".format(obj.split('/')[-1].replace('.blend', ''))), "w") as f:
#             f.write(out_str)

if __name__ == "__main__":
    main()