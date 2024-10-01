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
    parser.add_argument("--obj_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument('--N_FRAMES', type=int, default = 6)

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    DATASET = args.dataset_type
    N_FRAMES = args.N_FRAMES
    obj_count = 1


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


    #render_utils.add_plane((0,0,0), 10.0)


    scn = bpy.context.scene

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    # Add four lights
    light_locations = [(0, 0, 5), (5, 0, 0), (0, 5, 0), (-5, 0, 0)]  # Positions of the lights

    for i, location in enumerate(light_locations):
        bpy.ops.object.light_add(type='POINT', location=location)
        light = bpy.context.object
        light.data.energy = 100  # Adjust energy as needed
        light.name = f"PointLight_{i+1}"

    load_end_time = load_start_time = time.time()

    ################### HARDCODE
    # config["objects"] = []
    ################### HARDCODE
    
    box_2d_list = []
    obj_list = []
    pose_list = []
    obj_names = []
    
    # nothing is active, nothing is selected
    bpy.context.view_layer.objects.active = None

    for o in bpy.data.objects:
        o.select_set(False)
        
    load_start_time = time.time()
        
    obj_path = args.obj_path

    obj = render_utils.load_obj(scn, obj_path, args.dataset_type)
    load_end_time = time.time()
    obj.select_set(True)  
    bpy.context.view_layer.objects.active = obj
       
        # clear normals
    bpy.ops.mesh.customdata_custom_splitnormals_clear()

        # recompute normals
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()
        
    ### rescaling object to fit in unit cube
    #vertices = np.array([v.co for v in obj.data.vertices])
    obj.scale = (0.1843, 0.1843, 0.1843)#obj.scale * 0.5 / np.max(np.abs(vertices))

    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)
    

    ### calculate z_shift to put object on plane surface
    pose_mat = np.eye(4,4) # empty variable where we will store the pose matrix

        # compute how much to shift the object up so it appears on top of the surface
    vertices = np.array([v.co for v in obj.data.vertices])
    vertices_w_co = (vertices.T).T
    z_shift = np.abs(vertices_w_co[:,2].min())

    #pose_mat[2,-1] = z_shift
        
    #obj.matrix_world = Matrix(pose_mat)

    _, obj_height = get_2d_world_bbox_of_object(obj)

    min_bounds, max_bounds = get_scene_bounds(scn)

    angles = np.linspace(0.0, 2*np.pi, N_FRAMES)

    radii = 1.2
    height = 0.6 # obj_height
    x_cam_co = radii*np.cos(angles)
    y_cam_co = radii*np.sin(angles)
    z_cam_co = [height]*N_FRAMES
    camera_positions = np.stack([x_cam_co, y_cam_co, z_cam_co]).T

    # Choose camera track-to point
    track_to_points = np.zeros((N_FRAMES, 3))

    track_to_points[:,-1] = 0.3#obj_height/2

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

    
    output_dirpath = args.output_path
    try:
        render_utils.do_compositing(output_dirpath, obj_list)

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


    except:
        pass
    

if __name__ == "__main__":
    main()