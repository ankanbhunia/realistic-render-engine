'''
feakilename = "/home/sstojanov3/develop/CRIBpp_generic/rendering/generate.py"
exec(compile(open(filename).read(), filename, 'exec'))                                      
'''

import bpy
import numpy as np
import colorsys
import json
import os
import sys
import argparse
import time
import random, math
from mathutils import Matrix, Vector
from mathutils import Vector
import traceback
### ugly but necessary because of Blender's Python
fpath = bpy.data.filepath
root_path = '/'.join(fpath.split('/')[:-2])
blend_file_dir_path = os.path.join(root_path, "common")
python_file_dir_path = os.path.join(root_path, "rendering")

sys.path.append(blend_file_dir_path)
sys.path.append(python_file_dir_path)
os.chdir('renderings/data_generation/scene_config_generation')

import render_utils

def random_wave_function(min, max, n_frames):

  a, b = min, max
  phase = random.uniform(0, math.pi)
  first_half = [((b-a)/2)*math.sin(i+phase)+(a+b)/2 for i in np.linspace(0,math.pi*random.uniform(0.1, 2), n_frames//2+1)]
  first_half += first_half[::-1]

  return first_half[:n_frames]

def get_scene_bounds(scene, exclude_object_name = 'floor_object'):
    # Initialize min and max vectors with large and small values
    min_bound = Vector((float('inf'), float('inf'), float('inf')))
    max_bound = Vector((-float('inf'), -float('inf'), -float('inf')))

    # Iterate through all objects in the scene
    for obj in scene.objects:
        # Check if the object is visible, has geometry, and is not the excluded object
        if obj.type == 'MESH' and obj.visible_get() and obj.name != exclude_object_name:
            # Get the world matrix of the object
            obj_matrix = obj.matrix_world

            # Iterate through all vertices of the object
            for vertex in obj.data.vertices:
                # Get the vertex coordinates in world space
                world_vertex = obj_matrix @ vertex.co

                # Update min and max bounds
                min_bound.x = min(min_bound.x, world_vertex.x)
                min_bound.y = min(min_bound.y, world_vertex.y)
                min_bound.z = min(min_bound.z, world_vertex.z)
                max_bound.x = max(max_bound.x, world_vertex.x)
                max_bound.y = max(max_bound.y, world_vertex.y)
                max_bound.z = max(max_bound.z, world_vertex.z)

    return min_bound, max_bound

def get_2d_world_bbox_of_object(obj):

    bounding_box = obj.bound_box

    bounding_box_world = [obj.matrix_world @ Vector(point[:]) for point in bounding_box]
    
    box_xys = []
    heights = []
    for point in bounding_box_world:
        box_xys.append(np.array(point)[:2])
        heights.append(point[2])
    
    return box_xys[::2], max(heights)

import numpy as np
from scipy.spatial.distance import cdist, pdist

def rectangle_distance(r1, r2):
    def interpolate_points(start_point, end_point, num_points):
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)  # Interpolation parameter between 0 and 1
            x = start_point[0] + t * (end_point[0] - start_point[0])
            y = start_point[1] + t * (end_point[1] - start_point[1])
            points.append((x, y))
        return points

    def get_points_on_sides(r, num_points):

        if np.dot(r[0]-r[1], r[1]-r[2])<0.0001:
            sides = [interpolate_points(r[0], r[1], num_points), 
                    interpolate_points(r[1], r[2], num_points),
                    interpolate_points(r[2], r[3], num_points),
                    interpolate_points(r[3], r[0], num_points)]
        else:
            sides = [interpolate_points(r[0], r[1], num_points), 
                    interpolate_points(r[1], r[3], num_points),
                    interpolate_points(r[3], r[2], num_points),
                    interpolate_points(r[2], r[0], num_points)]
        points = sum(sides, [])

        return points

    p1 = get_points_on_sides(r1, 5)
    p2 = get_points_on_sides(r2, 5)

    return (cdist(p1,p2)).min()


def main():
    ### load datagen params
    with open(os.path.join(python_file_dir_path, "data_generation_parameters.json")) as load_file:
        data_gen_params = json.load(load_file)
    
    cam_params = data_gen_params["camera_parameters"]
    render_params = data_gen_params["render_parameters"]
    
    # bpy.data.scenes[0].render.engine = "CYCLES"
    bpy.data.scenes[0].render.engine = "BLENDER_EEVEE"

    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        if d.type == "GPU":
            d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])
    
    ### read arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--obj_paths_input", nargs='+')
    parser.add_argument("--output_path", type=str)
    parser.add_argument('--N_FRAMES', type=int, default = 30)

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    DATASET = args.dataset_type
    N_FRAMES = args.N_FRAMES
    obj_count = len(args.obj_paths_input)
    OBJ_SCALE_MIN = 0.40
    OBJ_SCALE_MAX = 0.41
    JITTERING = 0.01
    CAM_ANGLES = "FIXED UNIFORM" # OR "FIXED UNIFORM"
    hrd_paths = '../../../renderings/data_generation/common/assets/hdr'
    pbr_paths = '../../../renderings/data_generation/common/assets/pbr'

    def get_id_info(path, dataset_type):
        if dataset_type == "modelnet":
            category = path.split("/")[-3]
            obj_id = path.split("/")[-1][:-4]
            return category, obj_id

        if dataset_type == "ABC":
            obj_id = path.split("/")[-1]
            category = ''
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
                'OBJ_POSE_DIR', 
                category, 
                obj_id, 
                "pose_list.json"
            )
        try:
            with open(pose_json_path, "r") as f:
                pose_list = json.load(f)

            pose = np.random.choice(pose_list)

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

        pose = pose.tolist()
        
        return pose

        
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


    hdr_files = os.listdir(hrd_paths) 
    pbr_files = os.listdir(pbr_paths) 
    floor_pbr = np.random.choice(pbr_files)
    bg_hdr = np.random.choice(hdr_files) 
    env_strength = np.random.uniform(0.7, 0.9)

    objects = args.obj_paths_input #[obj]*(obj_count)
    object_poses = [pick_object_pose(obj) for obj in objects]
    object_scales = [np.random.uniform(OBJ_SCALE_MIN, OBJ_SCALE_MAX)]*(obj_count)

    render_utils.add_plane((0,0,0), 10.0)
    
    pbr_name = render_utils.add_PBR(
        'floor_material', 
        floor_pbr,
        os.path.join(blend_file_dir_path, "assets", "pbr")
        )
    bpy.data.objects['floor_object'].data.materials.append(bpy.data.materials['floor_material'])
    
    hdr_name, map_node = render_utils.add_IBL(
            bg_hdr,
            os.path.join(blend_file_dir_path, "assets", "hdr"),
            env_strength
            )
   
    scn = bpy.context.scene
    
    load_end_time = load_start_time = time.time()

    ################### HARDCODE
    # config["objects"] = []
    ################### HARDCODE
    
    box_2d_list = []
    obj_list = []
    pose_list = []
    obj_names = []
    for obj_idx, obj_name in enumerate(objects):
    
        # nothing is active, nothing is selected
        bpy.context.view_layer.objects.active = None

        for o in bpy.data.objects:
            o.select_set(False)
        
        load_start_time = time.time()
        
        obj_path = obj_name
        obj = render_utils.load_obj(scn, obj_path, args.dataset_type)
        obj_names.append(obj)
        load_end_time = time.time()
        # make loaded object active and selected
        obj.select_set(True)  
        bpy.context.view_layer.objects.active = obj
       
        # clear normals
        bpy.ops.mesh.customdata_custom_splitnormals_clear()

        # recompute normals
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.editmode_toggle()
        
        ### rescaling object to fit in unit cube
        vertices = np.array([v.co for v in obj.data.vertices])
        obj.scale = obj.scale * 0.5 / np.max(np.abs(vertices))
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)
        
        ### rescaling object to actual desired scale
        obj.scale = [object_scales[obj_idx]]*3
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        ### calculate z_shift to put object on plane surface
        pose_mat = np.eye(4,4) # empty variable where we will store the pose matrix
        rot_mat = np.array(object_poses[obj_idx]) #3x3 pose rotation matrix
        pose_mat[:3,:3] = rot_mat #set the rotation part of the pose matrix
        
        # compute how much to shift the object up so it appears on top of the surface
        vertices = np.array([v.co for v in obj.data.vertices])
        vertices_w_co = (rot_mat @ vertices.T).T
        z_shift = np.abs(vertices_w_co[:,2].min())

        ### set location in the pose matrix
        pose_mat[0,-1] = 0#obj_dict['obj_location'][0]
        pose_mat[1,-1] = 0#obj_dict['obj_location'][1]
        pose_mat[2,-1] = z_shift
        
        ### set pose matrix

        obj.matrix_world = Matrix(pose_mat)

        obj_list.append(obj)
        pose_list.append(pose_mat)
        

        #if args.dataset_type == 'ABC':

        #    mat_name = render_utils.add_configured_material(bpy.data, obj_dict['obj_color'])
        #    render_utils.assign_material(obj, mat_name)

    ### put objects in random location && iterate over a for loop until gets a desired layout
    
    min_value = -math.inf
    opt_obj_locations = None
    opt_means = None

    for multiplier_factor in [2,3]:

        for try_ in range(20):

            box_2d_list = []
            rect, hgts = get_2d_world_bbox_of_object(obj_list[0])
            #print (hgts)
            side1 = np.sqrt(((rect[0]-rect[1])**2).sum())
            side2 = np.sqrt(((rect[1]-rect[3])**2).sum())
            per_obj_area = side1*side2+0.05*hgts
            total_area_req = per_obj_area*len(obj_list)*multiplier_factor
            side_range = np.sqrt(total_area_req/4)
            print (side_range)
            obj_locations, means = pick_object_location(len(obj_list), -side_range, side_range, -side_range, side_range, max(side1, side2))
            print (obj_locations)
            if obj_locations == -1:
                continue
            for obj, pose_mat, loc in zip(obj_list, pose_list, obj_locations):
                pose_mat[0,-1] = loc[0]
                pose_mat[1,-1] = loc[1]
                obj.matrix_world = Matrix(pose_mat)
                box_2d_list.append(get_2d_world_bbox_of_object(obj)[0])
            min_rect_dist = []
            for rec1 in box_2d_list:
                for rec2 in box_2d_list:
                    if ((np.array(rec1) - np.array(rec2))**2).sum()>0:
                        min_rect_dist.append(rectangle_distance(rec1,rec2))
            print ('#######################')
            print (min(min_rect_dist))
            if min(min_rect_dist)>min_value:
                min_value = min(min_rect_dist)
                opt_obj_locations = obj_locations
                opt_means = means

        if opt_obj_locations is not None:
            break


    for obj, pose_mat, loc in zip(obj_list, pose_list, opt_obj_locations):
        pose_mat[0,-1] = loc[0]
        pose_mat[1,-1] = loc[1]
        obj.matrix_world = Matrix(pose_mat)

    ### camera pos & pose
    min_bounds, max_bounds = get_scene_bounds(scn)
    radius_max = max(abs(min_bounds[0]), abs(min_bounds[1]), abs(max_bounds[0]), abs(max_bounds[1]))+0.07*len(obj_list)
    height_max = max_bounds[2]+radius_max/3

    radius = random.uniform(radius_max, radius_max )
    height = random.uniform(height_max, height_max )

    
    if CAM_ANGLES == "FIXED UNIFORM":
        angles = np.linspace(0.0, 2*np.pi, N_FRAMES)
    if CAM_ANGLES == "RANDOM UNIFORM":
        angles = np.random.uniform(low=0.0, high=2*np.pi, size=N_FRAMES)
    radii = random_wave_function(min = radius, max = radius+0.1, n_frames =  N_FRAMES) #np.random.uniform(low=radius-0.2, high=radius+0.2, size=N_FRAMES)
    x_cam_co = radii*np.cos(angles)
    y_cam_co = radii*np.sin(angles)
    z_cam_co = random_wave_function(min = height, max = height+random.uniform(0, 0.4), n_frames = N_FRAMES) #np.random.uniform(height-0.1, height+0.1, N_FRAMES)
    camera_positions = np.stack([x_cam_co, y_cam_co, z_cam_co]).T

    # Choose camera track-to point
    track_to_points = np.zeros((N_FRAMES, 3))

    print (opt_means)
    track_to_points[:,0:2] = track_to_points[:,0:2] + opt_means.reshape(1,-1) + \
        np.random.uniform(-JITTERING, JITTERING, N_FRAMES*2).reshape(N_FRAMES,-1)
    
    track_to_points[:,2] = hgts/2
    camera_dict = {
            "track_to_point":track_to_points.tolist(),
            "positions":camera_positions.tolist()
        }

    bpy.context.view_layer.update()
    ## rendering settings
    render_utils.apply_settings(render_params, scn)
    
    # where to output renders
    #####################################################
    obj_list = [obj.name for obj in bpy.data.objects]

    obj_paths = [None] + objects

    print (obj_list, obj_paths)
    
    output_dirpath = args.output_path
    try:
        render_utils.do_compositing(output_dirpath, obj_list)
        print ("$$$$$$$$$$")
        # Add camera
        cam = render_utils.add_camera(cam_params)
        scn.camera = cam
        constraint_object, parent_object = render_utils.constrain_camera(cam, location=(0,0,0.0))
        scn.frame_start = 0

        # set camera position
        cam_positions = camera_dict['positions']
        n_frames = len(cam_positions)
        scn.frame_end = n_frames
        cam_rotations = []
        
        for i in np.arange(n_frames):
            scn.frame_set(i)
            parent_object.location = cam_positions[i]
            parent_object.keyframe_insert(data_path='location', frame = i)
            constraint_object.location = camera_dict['track_to_point'][-1]

            bpy.context.view_layer.update()
            cam_rotations.append(render_utils.get_3x4_RT_matrix_from_blender(cam))
        
        K = render_utils.get_calibration_matrix_K_from_blender(cam.data)
        ###########################################
        # render scene
        scn.frame_set(0)
        
        for i in np.arange(n_frames):
            scn.frame_set(i)
            bpy.ops.render.render()
        ##########################################



        # prepare and output metadata
        metadata = {}
        
        metadata['objects'] = []
        metadata['camera'] = {
            "poses":[],
            "K":np.array(K).tolist()
        }
        metadata["scene"] = {
            "floor_pbr":floor_pbr,
            "background_hdr":bg_hdr
        }
       
        for i in range(n_frames):
            metadata['camera']['poses'].append(
                {
                    "rotation":np.array(cam_rotations[i]).tolist(),
                }
            )
        print (obj_list,obj_paths)
        print (obj_names,objects)
        print (len(obj_names),len(objects))
        for obj, obj_path_name in zip(obj_names,objects):
            # obj = bpy.data.objects[obj_name]
            
            metadata['objects'].append(
                {
                    "name":obj.name,
                    "path":obj_path_name,
                    "rotation_matrix":np.array(obj.matrix_world).tolist(),
                    "rotation_euler":np.array(obj.rotation_euler).tolist(),
                    "location":np.array(obj.location).tolist(),
                    "scale":np.array(obj.scale).tolist(),
                    "bbox":[np.array(obj.location).tolist(), np.array(obj.dimensions).tolist()]
                }
            )
       
        metadata['rendering_info'] = {
            "object_loading_time":load_end_time - load_start_time
        }
        
        meta_str = json.dumps(metadata, indent=True)

        with open(os.path.join(output_dirpath, "scene3d.metadata.json"), "w") as f:
            f.write(meta_str)

    except Exception as e:
        traceback.print_exc()
        print(f'err:  Exception: {e}')

    print (radius, height)

if __name__ == "__main__":
    main()