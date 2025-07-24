import bpy
import sys
import os
from mathutils import Matrix
import math,copy,shutil
import numpy as np
import argparse
import json, mathutils
import random, statistics
import bpy_extras

# Ugly but necessary because of Blender's Python

### ugly but necessary because of Blender's Python
fpath = bpy.data.filepath
root_path = '/'.join(fpath.split('/')[:-2])
print (root_path)

sys.path.append(os.path.join(root_path, "utils"))
print (os.path.join(root_path, "utils"))
import render_utils

def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # For Blender's internal random functions, if any, they might need specific settings
        # For example, for Cycles noise seed:
        # bpy.context.scene.cycles.seed = seed
        # For general animation/physics:
        # bpy.context.scene.frame_set(bpy.context.scene.frame_current) # This might reset some simulations
        # For specific modifiers or tools, you might need to set their seed properties if they exist.
        print(f"Random seed set to: {seed}")
    else:
        print("No seed provided, using default random behavior.")

# Function to disable shadows for all lights
def disable_light_shadows():
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.data.use_shadow = False  # Disable shadows for the light

# Function to disable shadow casting for all objects in Cycles
def disable_object_shadows():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Disable shadow casting for Cycles
            obj.cycles_visibility.shadow = False

def link_nodes(tree, from_node, from_socket_name, to_node, to_socket_name):
    tree.links.new(from_node.outputs[from_socket_name], to_node.inputs[to_socket_name])


def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K

def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],),
        (0,0,0,1)
         ))
    return RT

def get_K():

    # Select the camera
    camera = bpy.data.objects['Camera']  # Replace 'Camera' with the name of your camera if it's different

    # Get camera data
    cam_data = camera.data

    # Render resolution
    scene = bpy.context.scene
    resolution_x = scene.render.resolution_x
    resolution_y = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width = cam_data.sensor_width
    sensor_height = cam_data.sensor_height if cam_data.sensor_fit == 'VERTICAL' else sensor_width

    # Focal length and principal point
    focal_length = cam_data.lens
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    principal_x = resolution_x * scale / 2
    principal_y = resolution_y * scale / 2

    # Intrinsic matrix calculation
    f_x = (resolution_x * scale) / sensor_width * focal_length
    f_y = (resolution_y * scale * pixel_aspect_ratio) / sensor_height * focal_length

    intrinsic_matrix = np.array([[f_x, 0, principal_x],
                                [0, f_y, principal_y],
                                [0, 0, 1]])

    return intrinsic_matrix

from mathutils import Vector

def is_inside(point, obj):
    """
    Checks if a point is inside a mesh object.

    Args:
        point (mathutils.Vector or tuple): The world-space coordinate to check.
        obj (bpy.types.Object): The mesh object to check against.

    Returns:
        bool: True if the point is inside the object, False otherwise.
    """
    # Ensure the point is a Vector
    point_world = Vector(point)

    # 1. Get the object's evaluated dependency graph to handle modifiers
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)

    # 2. Get the inverse of the object's world matrix to transform the point
    # into the object's local coordinate space.
    matrix_inv = obj_eval.matrix_world.inverted()
    point_local = matrix_inv @ point_world

    # 3. Cast a ray from the point in an arbitrary direction (e.g., positive Z)
    # The direction and distance are defined in the object's local space.
    direction = Vector((0, 0, 1))
    # We need a long enough ray to be sure it exits the object.
    # The object's bounding box diagonal is a safe bet for distance.
    ray_distance = obj_eval.dimensions.length 

    # Perform the raycast from our local point
    # Note: obj.ray_cast returns (hit, location, normal, face_index)
    # The third argument is the ray's length.
    intersections = 0
    start_point = point_local
    
    # Keep casting rays until we no longer hit anything
    while True:
        hit, loc, norm, face_idx = obj_eval.ray_cast(start_point, direction, distance=ray_distance)
        if not hit:
            break # No more intersections found
        
        intersections += 1
        
        # To continue casting, we need to move the start point slightly past
        # the last hit location to avoid hitting the same face repeatedly.
        start_point = loc + direction * 0.0001


    # 4. An odd number of intersections means the point is inside.
    return intersections % 2 == 1

def compute_visibility_score(cam_obj, obj_to_check, vertices_to_check):
    """
    Calculates the visibility score of an object's vertices from a camera.

    This function checks two things:
    1. Is the vertex within the camera's view frustum?
    2. Is there any other object blocking the line of sight from the camera to the vertex?

    Args:
        cam_obj (bpy.types.Object): The camera object.
        obj_to_check (bpy.types.Object): The mesh object to check.
        vertices_to_check (list of int): A list of vertex indices to evaluate.

    Returns:
        tuple: A tuple containing:
            - score (float): The visibility score (0.0 to 1.0), considering occlusions.
            - in_view_score (float): The fraction of vertices within the camera's view frustum (0.0 to 1.0).
            - visible_vertices (list): A list of indices of the visible vertices.
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Get the evaluated object to account for modifiers
    obj_eval = obj_to_check.evaluated_get(depsgraph)
    
    # World matrix of the object to check
    matrix_world = obj_to_check.matrix_world
    
    # Get camera's location and direction in world space
    cam_location = cam_obj.matrix_world.translation
    
    visible_vertices = []
    in_view_vertices = []
    total_vertices = len(vertices_to_check)

    if total_vertices == 0:
        return 0.0, 0.0, []

    for v_index in vertices_to_check:
        # Get vertex coordinate in world space
        vert_co_world = matrix_world @ obj_to_check.data.vertices[v_index].co
        
        # --- 1. Check if the vertex is within the camera's view frustum ---
        co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, vert_co_world)
        
        # co_ndc is in normalized device coordinates (0-1 for x and y)
        # The Z value is the distance from the camera.
        # Check if vertex is in front of the camera and within the view frame
        if (0.0 < co_ndc.x < 1.0 and 0.0 < co_ndc.y < 1.0 and co_ndc.z > 0):
            in_view_vertices.append(v_index) # This vertex is within the view frustum
            
            # --- 2. Raycast from camera to vertex to check for occlusions ---
            direction = (vert_co_world - cam_location).normalized()
            
            # Perform the raycast
            hit, location, normal, index, hit_obj, matrix = scene.ray_cast(depsgraph, cam_location, direction)
            
            if hit:
                # An object was hit. Check if the hit object is the object we are evaluating
                # and if the hit location is very close to the actual vertex location.
                # This ensures that the ray didn't hit another part of the same object farther away.
                distance_to_hit = (location - cam_location).length
                distance_to_vert = (vert_co_world - cam_location).length
                
                if abs(distance_to_hit - distance_to_vert) < 0.001:
                    # The first thing hit was our target vertex, so it's visible
                    visible_vertices.append(v_index)
            else:
                # No object was hit, which shouldn't happen if the vertex is in view,
                # but we can consider it visible.
                visible_vertices.append(v_index)
            
    score = len(visible_vertices) / total_vertices
    return score

def sample_camera_positions(
    center,
    radius,
    n_views=20,
    azimuth_range=(0, 360),
    elevation_range=(-30, 60),
    distance_multiplier=(1.5, 2.5),
):
    camera_positions = []

    for _ in range(n_views):
        azimuth_deg = random.uniform(*azimuth_range)
        elevation_deg = random.uniform(*elevation_range)
        dist = radius * random.uniform(*distance_multiplier)

        # Convert spherical to Cartesian coordinates
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)

        x = dist * math.cos(elevation) * math.cos(azimuth)
        y = dist * math.cos(elevation) * math.sin(azimuth)
        z = dist * math.sin(elevation)

        cam_location = center + mathutils.Vector((x, y, z))
        camera_positions.append(cam_location)

    return camera_positions

def get_bbox_of_the_scene(obj, attribute_name = "visible_target_ids", full = False):

    indices = [0,1] if full else [1]
    attr = obj.data.attributes.get(attribute_name)

    if not attr or attr.domain != 'POINT':
        raise ValueError(f"Attribute {attribute_name} not found or not per-vertex")

    # Collect vertices with part_id == 1
    verts_with_part_id_1 = [
        obj.matrix_world @ v.co
        for i, v in enumerate(obj.data.vertices)
        if attr.data[i].value in indices
    ]

    if not verts_with_part_id_1:
        raise ValueError(f"No vertices found with {attribute_name}")

    # Compute bounding box (min and max in x, y, z)
    min_corner = mathutils.Vector((min(v.x for v in verts_with_part_id_1),
                                min(v.y for v in verts_with_part_id_1),
                                min(v.z for v in verts_with_part_id_1)))

    max_corner = mathutils.Vector((max(v.x for v in verts_with_part_id_1),
                                max(v.y for v in verts_with_part_id_1),
                                max(v.z for v in verts_with_part_id_1)))

    verts_indices = [
        i
        for i, v in enumerate(obj.data.vertices)
        if attr.data[i].value in indices
    ]

    return (min_corner, max_corner), verts_indices

def render_mesh(input_json_path, obj_dir, mode=['lineart']):
    

    # Load object from input_json
    with open(input_json_path, 'r') as f:
        json_data = json.load(f)

    seed = json_data.get("seed", None) 

    set_random_seed(seed)
    
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPTIX' for newer NVIDIA GPUs
    bpy.context.scene.cycles.device = 'GPU'
    # Set up all available GPU devices
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences

    # Attempt to enable all GPU devices
    for device in cprefs.get_devices()[0]:
        device.use = True

    os.makedirs(obj_dir, exist_ok = True)

    bpy.context.scene.render.engine = 'CYCLES'

    bpy.context.scene.cycles.use_denoising = True

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    def vec_cos_theta(vec_a, vec_b):
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)    
        len_a = np.linalg.norm(vec_a)
        len_b = np.linalg.norm(vec_b)
        return (np.dot(vec_a, vec_b) / ( len_a * len_b ))


    def vec_to_euler(vec):
        y_cam_dir = copy.deepcopy(vec)
        y_cam_dir[1] = 0
        x_cam_dir = copy.deepcopy(vec)
        x_cam_dir[0] = 0
        z_cam_dir = copy.deepcopy(vec)
        z_cam_dir[2] = 0
        x_axis = np.array([1,0,0])
        y_axis = np.array([0,1,0])
        z_axis = np.array([0,0,1])
        x_rot_degree = math.acos(vec_cos_theta( vec, -z_axis ))
        z_rot_degree = math.acos(vec_cos_theta( z_cam_dir, y_axis ))
        if vec[0] > 0:
            z_rot_degree = -z_rot_degree
        if np.isnan(z_rot_degree):
            z_rot_degree = 0
        return [ x_rot_degree, 0, z_rot_degree ]
    


    scn = bpy.context.scene
    model, _ = render_utils.load_obj_v2(scn, json_data, add_materials = False)
    
    if model is None:
        sys.exit(0)

    ### rescaling object to fit in unit cube
    vertices = np.array([v.co for v in model.data.vertices])
    model.scale = model.scale * 0.5 / np.max(np.abs(vertices))
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    image_width, image_height = json_data["render_parameters"]['resolution_x'], json_data["render_parameters"]['resolution_y']
    # Set up the rendering parameters
    render_resolution = (image_width, image_height)

    scene = bpy.context.scene
    if scene.world is None:
        # create a new world
        new_world = bpy.data.worlds.new("New World")
        new_world.use_sky_paper = True
        scene.world = new_world
    bg_color = (0, 0, 0)  # Set RGB color values between 0.0 and 1.0
    world = bpy.context.scene.world
    world.cycles_visibility.diffuse = True
    world.use_nodes = False  # Disable the use of nodes
    world.color = bg_color

    # Create a new plane
    bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=(0, 0, 0))

    # Name the plane
    plane = bpy.context.object
    plane.name = "Floor"

    # Position the plane below the existing object
    # Assuming the existing object is named "YourObjectName"
    obj = model
    
    if obj:
        # Move the plane to just below the object
        # Calculate the bounding box of the object to determine its lowest point
        obj_min_z = min((obj.matrix_world @ vert.co).z for vert in obj.data.vertices)
        plane.location.z = obj_min_z  # Adjust the floor position slightly below the object

        # Optional: Scale the plane to be larger than the object
        plane.scale = (1, 1, 1)  # Adjust scaling as needed

    else:
        print(f"Object named '{object_name}' not found. Ensure the object name is correct.")

    scene = bpy.context.scene

    ### disable shadow
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Disable shadow casting
            obj.cycles_visibility.shadow = False

            # Disable self-shadowing
            if 'Cycles' in bpy.context.scene.render.engine:
                for mat in obj.data.materials:
                    if mat and mat.use_nodes:
                        bsdf = mat.node_tree.nodes.get('Principled BSDF')
                        if bsdf:
                            bsdf.inputs['Alpha'].default_value = 1  # Ensure transparency to avoid shadowing

    # Load camera sampling parameters from json_data
    N_FRAMES = json_data.get("num_views", 8)
    AZIMUTH_RANGE = tuple(json_data.get("azimuth_range", (0, 360)))
    ELEVATION_RANGE = tuple(json_data.get("elevation_range", (0, 60)))
    DISTANCE_MULTIPLIER = tuple(json_data.get("distance_multiplier", (1.5, 2.5)))
    CAMERA_SEARCH_SPACE = json_data.get("camera_search_space", 30)
    VISIBILTY_CHECK_ENABLED = json_data.get("visibility_check_enabled", True)
    JITTERING = 0.01 # Define JITTERING here or get from json_data if available

    # Get bounding box and vertex indices for visibility check
    (min_corner, max_corner), _ = get_bbox_of_the_scene(obj, attribute_name = "camera_target_ids")
    _, verts_indices = get_bbox_of_the_scene(obj, attribute_name = "visible_target_ids")

    bbox_center = (min_corner + max_corner) / 2
    bbox_size = max_corner - min_corner
    bbox_radius = bbox_size.length / 2  # half the diagonal

    # Add camera and constrain it
    json_cam_params = json_data.get("camera_parameters", {})
    cam_params = {
        "type": json_cam_params.get("type", "PERSP"),
        "clip_start": json_cam_params.get("clip_start", 0.1),
        "clip_end": json_cam_params.get("clip_end", 100.0),
        "focal_length": json_cam_params.get("focal_length", 35), # Default focal length
        "sensor_width": json_cam_params.get("sensor_width", 32), # Default sensor width
        "sensor_height": json_cam_params.get("sensor_height", 18) # Default sensor height
    }
    cam = render_utils.add_camera(cam_params)
    scn.camera = cam
    constraint_object, parent_object = render_utils.constrain_camera(cam, location=(0,0,0.0))

    rendered_camera_data = []

    initial_camera_positions = sample_camera_positions(
        bbox_center, bbox_radius, n_views=CAMERA_SEARCH_SPACE,
        azimuth_range=AZIMUTH_RANGE, elevation_range=ELEVATION_RANGE,
        distance_multiplier=DISTANCE_MULTIPLIER
    )
    
    if VISIBILTY_CHECK_ENABLED:
        candidate_cameras = []
        for i, current_cam_position in enumerate(initial_camera_positions):
            # Set camera position and track-to point for evaluation
            parent_object.location = current_cam_position
            current_track_to_point = bbox_center #+ np.random.uniform(-JITTERING, JITTERING, 3) # Use bbox_center as base
            constraint_object.location = current_track_to_point
            bpy.context.view_layer.update()

            # Obtain visibility score for each camera. If is_inside is True, assign visibility score to zero.
            visibilty_score = compute_visibility_score(cam, obj, verts_indices)
            
            # Derive verts_indices_inv and calculate visibilty_score_2
            all_verts_indices = [v.index for v in obj.data.vertices]
            verts_indices_inv = [idx for idx in all_verts_indices if idx not in verts_indices]
            visibilty_score_2 = compute_visibility_score(cam, obj, verts_indices_inv)
            
            # Take the final visibilty_score as the minimum of both
            visibilty_score = min(visibilty_score, visibilty_score_2)

            if is_inside(current_cam_position, obj):
                visibilty_score = 0.0
                print(f"Camera {i}: Inside object, visibility score set to 0.")
            
            print(f"Camera {i}: Visibility Score = {visibilty_score:.4f}")

            # Store candidate camera data
            candidate_cameras.append({
                "position": current_cam_position,
                "track_to_point": current_track_to_point,
                "visibility_score": visibilty_score,
                "K": get_calibration_matrix_K_from_blender(cam.data),
                "RT": get_3x4_RT_matrix_from_blender(cam),
            })

            # Sort candidate cameras by visibility score in descending order
            candidate_cameras.sort(key=lambda x: x["visibility_score"], reverse=True)

            # Calculate the number of cameras for the top 30%
            top_k_percent_count = max(N_FRAMES, int(len(candidate_cameras) * 0.2)) # Ensure at least 1 camera if list is not empty

            # Select the top 30% of cameras
            top_cameras = candidate_cameras[:top_k_percent_count]
            
            # Randomly sample N_FRAMES from top_cameras
            if len(top_cameras) > N_FRAMES:
                cameras_to_render = random.sample(top_cameras, N_FRAMES)
            else:
                cameras_to_render = candidate_cameras[:N_FRAMES] # Use all available top cameras if less than N_FRAMES
            
            print(f"Selected {len(cameras_to_render)} cameras for rendering from top {top_k_percent_count} candidates (after sorting and random sampling).")
    else: # VISIBILTY_CHECK_ENABLED is False
        print("Visibility check disabled. Using initial camera positions for rendering.")
        cameras_to_render = []
        # When visibility check is disabled, we still need to respect the seed for sampling
        # from initial_camera_positions if N_FRAMES is less than the total.
        # First, ensure initial_camera_positions is consistent if seed is set.
        # The sample_camera_positions function already uses random.uniform, which is seeded.
        
        # If we need to select a subset, we should sample from initial_camera_positions
        # to maintain consistency with the seed.
        # if len(initial_camera_positions) > N_FRAMES:
        #     initial_camera_positions = random.sample(initial_camera_positions, N_FRAMES)

        for i, current_cam_position in enumerate(initial_camera_positions):

            if is_inside(current_cam_position, obj):
                pass
            else:
                parent_object.location = current_cam_position
                current_track_to_point = bbox_center #+ np.random.uniform(-JITTERING, JITTERING, 3)
                constraint_object.location = current_track_to_point
                bpy.context.view_layer.update() # Update scene to get correct K and RT
                
                cameras_to_render.append({
                    "position": current_cam_position,
                    "track_to_point": current_track_to_point,
                    "visibility_score": 1.0, # Assign a dummy score as no check is performed
                    "K": get_calibration_matrix_K_from_blender(cam.data),
                    "RT": get_3x4_RT_matrix_from_blender(cam),
                })
        
        # Respect N_FRAMES even when visibility check is disabled
        if len(cameras_to_render) > N_FRAMES:
            cameras_to_render = random.sample(cameras_to_render, N_FRAMES)
        
        print(f"Selected {len(cameras_to_render)} cameras for rendering (visibility check disabled).")


    # --- Hide vertices based on per_vertex_visibility attribute ---
    obj = model # The loaded model is the active object
    if obj.data.attributes.get("visibility"):
        # --- 1. Switch to Edit Mode ---
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type="VERT")
        mesh = obj.data
        visibility_attr = mesh.attributes["visibility"]

        vertex_group = obj.vertex_groups.new(name='VerticesToHide')
        vertices_to_hide_indices = []
        for i, vertex in enumerate(mesh.vertices):
            if visibility_attr.data[i].value == 0.0:
                vertices_to_hide_indices.append(i)

        bpy.ops.object.mode_set(mode='OBJECT')
        vertex_group.add(vertices_to_hide_indices, 1.0, 'REPLACE')
        mask_modifier = obj.modifiers.new(name="HideMask", type='MASK')
        mask_modifier.vertex_group = vertex_group.name
        mask_modifier.invert_vertex_group = True
        mask_modifier.show_render = True
        mask_modifier.show_viewport = True # Also hide in Object Mode viewport
        print(f"Added Mask modifier to '{obj.name}' to hide vertices in group '{vertex_group.name}' during render.")


    rendered_frames_count = 0
    for i, cam_data in enumerate(cameras_to_render):
        scn.frame_set(rendered_frames_count) # Set frame for rendering
        
        # Ensure current camera position is set and keyframed for this frame
        parent_object.location = cam_data["position"]
        # parent_object.keyframe_insert(data_path='location', frame=rendered_frames_count) # Keyframing not needed for still renders
        
        constraint_object.location = cam_data["track_to_point"]
        # constraint_object.keyframe_insert(data_path='location', frame=rendered_frames_count) # Keyframing not needed for still renders
        
        # Set up rendering settings
        bpy.context.scene.render.resolution_x = render_resolution[0]
        bpy.context.scene.render.resolution_y = render_resolution[1]

        # Set output path for rendered image
        file_pointer = f'{rendered_frames_count:04d}'
        output_path = os.path.join(obj_dir, f'info_{file_pointer}.png')

        ### depth rendering ###
        bpy.context.scene.cycles.samples = 1
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        for n in tree.nodes:
            tree.nodes.remove(n)

        ##### .png file, used for controlnet ####

        # render_layers_node = tree.nodes.new('CompositorNodeRLayers')
        # file_output_node = tree.nodes.new('CompositorNodeOutputFile')
        # normalize_node = tree.nodes.new('CompositorNodeNormalize')
        # invert_node = tree.nodes.new('CompositorNodeInvert')
        # map_value_node = tree.nodes.new('CompositorNodeMapValue')

        # render_layers_node.location = 0, 0
        # normalize_node.location = 200, 0
        # invert_node.location = 400, 0
        # map_value_node.location = 600, 0
        # file_output_node.location = 800, 0

        # file_output_node.base_path = obj_dir
        # file_output_node.file_slots[0].path = f'depth_{file_pointer}_' # Corrected filename

        # links.new(render_layers_node.outputs['Depth'], normalize_node.inputs[0])
        # links.new(normalize_node.outputs[0], invert_node.inputs[1])
        # links.new(invert_node.outputs[0], map_value_node.inputs[0])
        # links.new(map_value_node.outputs[0], file_output_node.inputs[0])

        # ##### .exr file, more accurate ####

        # render_layers_node = tree.nodes.new('CompositorNodeRLayers')
        # file_output_node = tree.nodes.new('CompositorNodeOutputFile')
        # map_value_node = tree.nodes.new('CompositorNodeMapValue')

        # file_output_node.format.file_format = 'OPEN_EXR'
        # file_output_node.format.color_depth = '32'
        # file_output_node.format.color_mode = 'RGB'

        # links.new(render_layers_node.outputs['Depth'], map_value_node.inputs[0])
        # links.new(map_value_node.outputs[0], file_output_node.inputs[0])

        # file_output_node.base_path = obj_dir
        # file_output_node.file_slots[0].path = f'depth_{file_pointer}_' # Corrected filename

        # Reset world settings to a neutral state before each mode's specific setup
        bpy.context.scene.world.use_nodes = False
        bpy.context.scene.world.color = (0, 0, 0) # Set to black initially, specific modes can change

        # Reset freestyle settings
        bpy.context.scene.render.use_freestyle = False
        # # Clear existing linesets
        # for ls in list(bpy.context.scene.view_layers[0].freestyle_settings.linesets):
        #     bpy.context.scene.view_layers[0].freestyle_settings.linesets.remove(ls)
        # # Clear existing linestyles
        # for ls in list(bpy.data.linestyles):
        #     bpy.data.linestyles.remove(ls)

        # Reset object visibility (plane)
        plane.hide_render = False # Default to visible, modes will adjust

        # Set Cycles exposure to a neutral value
        bpy.context.scene.cycles.film_exposure = 1.0 # Default exposure

        if 'depth' in mode:
            ##### .png file, used for controlnet ####
            render_layers_node = tree.nodes.new('CompositorNodeRLayers')
            file_output_node_png = tree.nodes.new('CompositorNodeOutputFile')
            normalize_node = tree.nodes.new('CompositorNodeNormalize')
            invert_node = tree.nodes.new('CompositorNodeInvert')
            map_value_node_png = tree.nodes.new('CompositorNodeMapValue')

            render_layers_node.location = 0, 0
            normalize_node.location = 200, 0
            invert_node.location = 400, 0
            map_value_node_png.location = 600, 0
            file_output_node_png.location = 800, 0

            file_output_node_png.base_path = obj_dir
            file_output_node_png.file_slots[0].path = f'depth_{file_pointer}_' # Corrected filename
            file_output_node_png.file_slots[0].use_node_format = False
            file_output_node_png.format.file_format = 'PNG'
            file_output_node_png.format.color_mode = 'BW'
            file_output_node_png.format.color_depth = '8'

            links.new(render_layers_node.outputs['Depth'], normalize_node.inputs[0])
            links.new(normalize_node.outputs[0], invert_node.inputs[1])
            links.new(invert_node.outputs[0], map_value_node_png.inputs[0])
            links.new(map_value_node_png.outputs[0], file_output_node_png.inputs[0])

            ##### .exr file, more accurate ####
            render_layers_node_exr = tree.nodes.new('CompositorNodeRLayers')
            file_output_node_exr = tree.nodes.new('CompositorNodeOutputFile')
            map_value_node_exr = tree.nodes.new('CompositorNodeMapValue')

            file_output_node_exr.format.file_format = 'OPEN_EXR'
            file_output_node_exr.format.color_depth = '32'
            file_output_node_exr.format.color_mode = 'RGB'

            links.new(render_layers_node_exr.outputs['Depth'], map_value_node_exr.inputs[0])
            links.new(map_value_node_exr.outputs[0], file_output_node_exr.inputs[0])

            file_output_node_exr.base_path = obj_dir
            file_output_node_exr.file_slots[0].path = f'depth_exr_{file_pointer}_' # Corrected filename

            bpy.ops.render.render(write_still=True)

            # Clear nodes for next render pass
            for n in tree.nodes:
                tree.nodes.remove(n)

        if 'lineart' in mode:
            ### lineart rendering ###
            bpy.context.scene.cycles.samples = 100
            plane.hide_render = True
            bpy.context.scene.render.use_freestyle = True

            ### add point lights
            for rid in [270,90,0,180]:
                key_light = bpy.data.lights.new(name="Key Light", type='POINT')
                key_light.energy = 666  # Adjust the strength of the light
                key_light_obj = bpy.data.objects.new(name="KeyLight", object_data=key_light)
                scene.collection.objects.link(key_light_obj)
                key_light_obj.location = (5*math.cos(math.radians(rid)), 5*math.sin(math.radians(rid)), 5)   # Adjust the position of the light

            lineset = bpy.context.scene.view_layers[0].freestyle_settings.linesets.new("LineSet")
            lineset.select_by_visibility = True
            lineset.visibility = 'VISIBLE'
            linestyle = bpy.data.linestyles.new("LineStyle")
            lineset.linestyle = linestyle
            linestyle.color = (1, 1, 1)
            linestyle.thickness = 1
            bpy.context.view_layer.freestyle_settings.use_culling = True
            bpy.context.scene.world.color = (1, 1, 1)
            bpy.context.scene.render.film_transparent = False
            bpy.context.scene.use_nodes = True
            tree = bpy.context.scene.node_tree
            links = tree.links
            lineset.select_contour = True
            lineset.select_crease = False
            lineset.select_edge_mark = False
            lineset.select_material_boundary = False
            lineset.select_border = False



            # lineset.visibility = 'HIDDEN' 
            for node in tree.nodes:
                tree.nodes.remove(node)

                
            # Create new nodes
            render_layers = tree.nodes.new('CompositorNodeRLayers')
            rgb_to_bw = tree.nodes.new('CompositorNodeRGBToBW')
            threshold = tree.nodes.new('CompositorNodeMath')
            composite = tree.nodes.new('CompositorNodeComposite')

            # Configure threshold node
            threshold.operation = 'GREATER_THAN'
            threshold.inputs[1].default_value = 0.5  # Threshold to binarize

            # Link nodes
            links.new(render_layers.outputs['Image'], rgb_to_bw.inputs['Image'])
            links.new(rgb_to_bw.outputs['Val'], threshold.inputs[0])
            links.new(threshold.outputs['Value'], composite.inputs['Image'])

            # Position nodes for clarity (optional)
            render_layers.location = (0, 0)
            rgb_to_bw.location = (200, 0)
            threshold.location = (400, 0)
            composite.location = (600, 0)  

            bpy.context.scene.render.filepath = os.path.join(obj_dir, f'lineart_{file_pointer}.png')
            bpy.ops.render.render(write_still=True)

        if 'textureless' in mode:
            ### textureless rendering ###

            
            bpy.context.scene.cycles.samples = 100
            plane.hide_render = True # Show plane for textureless render
            bpy.context.scene.render.use_freestyle = False # Disable freestyle for textureless
            bpy.context.scene.world.color = (0., 0., 0.) # Very dark grey background
            bpy.context.scene.world.use_nodes = False # Ensure world nodes are not used
            bpy.context.scene.render.film_transparent = False # Ensure film is not transparent

            # Remove all existing lights
            bpy.ops.object.select_by_type(type='LIGHT')
            bpy.ops.object.delete()


            ### add point lights
            for rid in [270,90,0,180]:
                for light_height in [5, -5]:
                    key_light = bpy.data.lights.new(name="Key Light", type='POINT')
                    key_light.energy = 30  # Adjust the strength of the light
                    key_light_obj = bpy.data.objects.new(name="KeyLight", object_data=key_light)
                    scene.collection.objects.link(key_light_obj)
                    key_light_obj.location = (5*math.cos(math.radians(rid)), 5*math.sin(math.radians(rid)), light_height)   # Adjust the position of the light

            # # Ensure all objects have a simple diffuse material
            # for obj_item in bpy.data.objects:
            #     if obj_item.type == 'MESH':
            #         # Ensure the object has at least one material slot
            #         if not obj_item.data.materials:
            #             obj_item.data.materials.append(bpy.data.materials.new(name="NeutralMaterial"))
                    
            #         # Create or get the neutral material
            #         mat = bpy.data.materials.get("NeutralMaterial")
            #         if not mat:
            #             mat = bpy.data.materials.new(name="NeutralMaterial")
            #             mat.use_nodes = True
            #             principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
            #             if not principled_bsdf:
            #                 principled_bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
            #                 mat.node_tree.links.new(principled_bsdf.outputs['BSDF'], mat.node_tree.nodes['Material Output'].inputs['Surface'])
                        
            #             principled_bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1) # Neutral grey
            #             principled_bsdf.inputs['Roughness'].default_value = 0.8 # Make it less shiny
            #             principled_bsdf.inputs['Specular'].default_value = 0.0
            #             principled_bsdf.inputs['Metallic'].default_value = 0.0
            #             principled_bsdf.inputs['Sheen'].default_value = 0.0
            #             principled_bsdf.inputs['Clearcoat'].default_value = 0.0
            #             principled_bsdf.inputs['Transmission'].default_value = 0.0

            #         # Assign the neutral material to all slots of the mesh
            #         for slot in obj_item.data.materials:
            #             slot = mat
            #         if not obj_item.data.materials: # If no slots, append
            #             obj_item.data.materials.append(mat)

            # Clear nodes for textureless render
            for n in tree.nodes:
                tree.nodes.remove(n)
            
            # Create new nodes for textureless render
            render_layers = tree.nodes.new('CompositorNodeRLayers')
            composite = tree.nodes.new('CompositorNodeComposite')

            # Link nodes
            links.new(render_layers.outputs['Image'], composite.inputs['Image'])

            # Position nodes for clarity (optional)
            render_layers.location = (0, 0)
            composite.location = (200, 0)

            bpy.context.scene.render.filepath = os.path.join(obj_dir, f'textureless_{file_pointer}.png')
            bpy.ops.render.render(write_still=True)

            # Clear nodes for next render pass
            for n in tree.nodes:
                tree.nodes.remove(n)


        # lineset = bpy.context.scene.view_layers[0].freestyle_settings.linesets.new("LineSet")
        # lineset.select_by_visibility = True
        # lineset.visibility = 'VISIBLE'
        # linestyle = bpy.data.linestyles.new("LineStyle")
        # lineset.linestyle = linestyle
        # linestyle.color = (1, 1, 1)
        # linestyle.thickness = 1
        # bpy.context.view_layer.freestyle_settings.use_culling = True
        # bpy.context.scene.world.color = (1, 1, 1)
        # bpy.context.scene.render.film_transparent = False
        # bpy.context.scene.use_nodes = True
        # tree = bpy.context.scene.node_tree
        # links = tree.links
        # lineset.select_contour = True
        # lineset.select_crease = False
        # lineset.select_edge_mark = False
        # lineset.select_material_boundary = False
        # lineset.select_border = False



        # # lineset.visibility = 'HIDDEN' 
        # for node in tree.nodes:
        #     tree.nodes.remove(node)

            
        # # Create new nodes
        # render_layers = tree.nodes.new('CompositorNodeRLayers')
        # rgb_to_bw = tree.nodes.new('CompositorNodeRGBToBW')
        # threshold = tree.nodes.new('CompositorNodeMath')
        # composite = tree.nodes.new('CompositorNodeComposite')

        # # Configure threshold node
        # threshold.operation = 'GREATER_THAN'
        # threshold.inputs[1].default_value = 0.5  # Threshold to binarize

        # # Link nodes
        # links.new(render_layers.outputs['Image'], rgb_to_bw.inputs['Image'])
        # links.new(rgb_to_bw.outputs['Val'], threshold.inputs[0])
        # links.new(threshold.outputs['Value'], composite.inputs['Image'])

        # # Position nodes for clarity (optional)
        # render_layers.location = (0, 0)
        # rgb_to_bw.location = (200, 0)
        # threshold.location = (400, 0)
        # composite.location = (600, 0)  

        # bpy.context.scene.render.filepath = os.path.join(obj_dir, f'lineart_{file_pointer}.png')
        # bpy.ops.render.render(write_still=True)


        ### save camera matrix
        
        rendered_camera_data.append({
            "frame": rendered_frames_count,
            "K": np.array(cam_data["K"]).tolist(),
            "RT": np.array(cam_data["RT"]).tolist(),
            "visibility_score": cam_data["visibility_score"],
        })
            
        # with open(output_path.replace('.png', '.json'), 'w+') as f:
        #     json.dump(proj_matrix, f, sort_keys=True)

        rendered_frames_count += 1
        print(f"Rendered frame {rendered_frames_count}: Visibility Score = {cam_data['visibility_score']:.4f}")

    scn.frame_end = rendered_frames_count - 1 if rendered_frames_count > 0 else 0 # Adjust scene frame end to actual rendered frames

    # Save camera information to a JSON file
    camera_info_filepath = os.path.join(obj_dir, "camera_info.json")
    with open(camera_info_filepath, 'w') as f:
        json.dump(rendered_camera_data, f, indent=4)
    print(f"Saved camera information to {camera_info_filepath}")

    dir_name_prev = os.path.dirname(os.path.dirname(obj_dir+'/')) 
    # vertices = [vert.co for vert in model.data.vertices]
    # faces = [face.vertices for face in model.data.polygons]
    # np.save(dir_name_prev + '/verts.npy', np.array([[v.x, v.y, v.z] for v in vertices]))
    # np.save(dir_name_prev + '/faces.npy', np.array(faces))


if __name__ == '__main__':

    import argparse, sys
    parser = argparse.ArgumentParser(description='Render 3D models with different modes.')
    parser.add_argument('--input_json', type=str, help='Path to the input JSON file containing object data.')
    parser.add_argument('--out_path', type=str, help='Output directory for rendered images and data.')
    parser.add_argument('--mode', type=str, default='lineart', 
                        help='Single rendering mode. Can be "lineart", "textureless", "depth". Default is "lineart".')

    # Argument parsing
    argv = sys.argv
    try:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"
    except ValueError:
        argv = [] # No "--" found, so no custom arguments

    args = parser.parse_args(argv)
    
    # render_mesh now expects a list for 'mode', so wrap the single string argument in a list
    render_mesh(args.input_json, args.out_path, mode=[args.mode])
