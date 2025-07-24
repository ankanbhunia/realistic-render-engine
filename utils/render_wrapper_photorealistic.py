import bpy
import numpy as np
import json
import os
import sys
import argparse
import time
import random, math
from mathutils import Matrix, Vector
import traceback, mathutils
import bpy_extras
import statistics

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

### ugly but necessary because of Blender's Python
fpath = bpy.data.filepath
root_path = '/'.join(fpath.split('/')[:-2])
print (root_path)

sys.path.append(os.path.join(root_path, "utils"))

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

def draw_bounding_box(min_corner, max_corner, name="BoundingBox", color=(1.0, 0.0, 0.0, 1.0)):
    """
    Draws a 3D bounding box in Blender using min_corner and max_corner.
    """
    # Define the 8 vertices of the bounding box
    vertices = [
        (min_corner.x, min_corner.y, min_corner.z),
        (max_corner.x, min_corner.y, min_corner.z),
        (min_corner.x, max_corner.y, min_corner.z),
        (max_corner.x, max_corner.y, min_corner.z),
        (min_corner.x, min_corner.y, max_corner.z),
        (max_corner.x, min_corner.y, max_corner.z),
        (min_corner.x, max_corner.y, max_corner.z),
        (max_corner.x, max_corner.y, max_corner.z),
    ]

    # Define the 12 edges of the bounding box
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]

    # Define the 6 faces of the bounding box (optional, for solid box)
    faces = [
        (0, 1, 3, 2),  # Bottom face
        (4, 5, 7, 6),  # Top face
        (0, 1, 5, 4),  # Front face
        (2, 3, 7, 6),  # Back face
        (0, 2, 6, 4),  # Left face
        (1, 3, 7, 5),  # Right face
    ]

    # Create mesh and object
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)

    # Link object to the scene
    bpy.context.collection.objects.link(obj)

    # Create a material for the bounding box
    mat = bpy.data.materials.new(name="BBoxMaterial")
    mat.diffuse_color = color
    obj.data.materials.append(mat)

    return obj

def add_pbr_material(obj_name, material_name, texture_folder, scale=(1.0, 1.0, 1.0)):
    # Create new material
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    # Define texture types and their corresponding maps
    textures = {
        "Base Color": "color",       # Base Color (Albedo)
        "Diffuse": "diffuse",        # Diffuse map
        "Normal": "normal",          # Normal map
        "Roughness": "roughness",    # Roughness map
        "Displacement": "displacement", # Displacement map
        "Specular": "specular",      # Specular map
        "Opacity": "opacity",        # Opacity map
        "Metallic": "metallic",      # Metallic map
        "Height": "height",          # Height map
    }
    
    # Create the UV Map node and the Mapping node
    tex_coord_node = mat.node_tree.nodes.new(type="ShaderNodeTexCoord")
    mapping_node = mat.node_tree.nodes.new(type="ShaderNodeMapping")
    mapping_node.inputs["Scale"].default_value = scale  # Set the scale for the textures
    mapping_node.location = (-800, 0)
    tex_coord_node.location = (-1000, 0)
    
    # Connect UV output to the Mapping node
    mat.node_tree.links.new(mapping_node.inputs['Vector'], tex_coord_node.outputs['UV'])
    
    # Load textures from the folder and connect them to the BSDF shader
    for texture_type, texture_keyword in textures.items():
        texture_file = next((f for f in os.listdir(texture_folder) if texture_keyword in f), None)
        if texture_file:
            texture_path = os.path.join(texture_folder, texture_file)
            
            # Create an image texture node
            tex_image_node = mat.node_tree.nodes.new(type="ShaderNodeTexImage")
            tex_image_node.image = bpy.data.images.load(texture_path)
            tex_image_node.label = texture_type
            tex_image_node.location = (-400, 200)
            
            # Connect the Mapping node to the texture coordinate input
            mat.node_tree.links.new(tex_image_node.inputs['Vector'], mapping_node.outputs['Vector'])
            
            # Handle different texture types
            if texture_type == "Base Color":
                # Connect Base Color to Principled BSDF
                mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
            elif texture_type == "Diffuse":
                # Connect Diffuse to Principled BSDF (Base Color)
                mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
            elif texture_type == "Roughness":
                # Connect Roughness to Principled BSDF
                mat.node_tree.links.new(bsdf.inputs['Roughness'], tex_image_node.outputs['Color'])
                tex_image_node.image.colorspace_settings.name = 'Non-Color'
            elif texture_type == "Normal":
                # Add a Normal Map node and connect to Principled BSDF
                normal_map_node = mat.node_tree.nodes.new(type="ShaderNodeNormalMap")
                mat.node_tree.links.new(normal_map_node.inputs['Color'], tex_image_node.outputs['Color'])
                mat.node_tree.links.new(bsdf.inputs['Normal'], normal_map_node.outputs['Normal'])
                tex_image_node.image.colorspace_settings.name = 'Non-Color'
            elif texture_type == "Displacement":
                # Add a Displacement node and connect to Material Output
                displacement_node = mat.node_tree.nodes.new(type="ShaderNodeDisplacement")
                mat.node_tree.links.new(displacement_node.inputs['Height'], tex_image_node.outputs['Color'])
                displacement_output = mat.node_tree.nodes.get("Material Output").inputs['Displacement']
                mat.node_tree.links.new(displacement_output, displacement_node.outputs['Displacement'])
                tex_image_node.image.colorspace_settings.name = 'Non-Color'
            elif texture_type == "Specular":
                # Connect Specular to Principled BSDF
                mat.node_tree.links.new(bsdf.inputs['Specular'], tex_image_node.outputs['Color'])
                tex_image_node.image.colorspace_settings.name = 'Non-Color'
            elif texture_type == "Opacity":
                # Connect Opacity to Principled BSDF Alpha input
                mat.node_tree.links.new(bsdf.inputs['Alpha'], tex_image_node.outputs['Color'])
                tex_image_node.image.colorspace_settings.name = 'Non-Color'
                # Set material to be transparent
                mat.blend_method = 'BLEND'
            elif texture_type == "Metallic":
                # Connect Metallic to Principled BSDF
                mat.node_tree.links.new(bsdf.inputs['Metallic'], tex_image_node.outputs['Color'])
                tex_image_node.image.colorspace_settings.name = 'Non-Color'
            elif texture_type == "Height":
                # Add Height to the displacement node
                displacement_node = mat.node_tree.nodes.new(type="ShaderNodeDisplacement")
                mat.node_tree.links.new(displacement_node.inputs['Height'], tex_image_node.outputs['Color'])
                displacement_output = mat.node_tree.nodes.get("Material Output").inputs['Displacement']
                mat.node_tree.links.new(displacement_output, displacement_node.outputs['Displacement'])
                tex_image_node.image.colorspace_settings.name = 'Non-Color'
    
    # Assign the material to the object
    obj = bpy.data.objects.get(obj_name)
    if obj is not None and obj.type == 'MESH':
        if len(obj.data.materials) > 0:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

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


import bpy
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

# --- Example of how to use the function ---
# Assumes you have a 'Camera' and an object named 'Cube'
# and you want to check the visibility of all its vertices.

# cam = bpy.context.scene.camera
# obj = bpy.data.objects.get("Cube")

# if cam and obj:
#    # Get all vertex indices for the object
#    all_verts = list(range(len(obj.data.vertices)))
#
#    # Calculate the visibility
#    visibility_score, visible_verts_indices = compute_visibility_score(cam, obj, all_verts)
#
#    print(f"Visibility Score for '{obj.name}': {visibility_score:.2f}")
#    print(f"{len(visible_verts_indices)} of {len(all_verts)} vertices are visible.")
# else:
#    print("Please make sure you have a Camera and an object named 'Cube' in the scene.")

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

def main():

    ### read arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", type=str)
    parser.add_argument("--output_path", type=str)

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    
    with open(args.input_json, 'r') as f:
    # Load the data from the file into a Python object
        json_data = json.load(f)

    seed = json_data.get("seed", None) 
    set_random_seed(seed)

    cam_params = json_data["camera_parameters"]
    render_params = json_data["render_parameters"]
    
    # Load camera sampling parameters from json_data
    N_FRAMES = json_data.get("num_views", 8)
    # VISIBILITY_THRESHOLD will be calculated dynamically
    AZIMUTH_RANGE = tuple(json_data.get("azimuth_range", (0, 360)))
    ELEVATION_RANGE = tuple(json_data.get("elevation_range", (0, 60)))
    DISTANCE_MULTIPLIER = tuple(json_data.get("distance_multiplier", (1.5, 2.5)))
    CAMERA_SEARCH_SPACE = json_data.get("camera_search_space", 30)
    VISIBILTY_CHECK_ENABLED = json_data.get("visibility_check_enabled", True)

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

        
    OBJ_SCALE_MIN = 0.40
    OBJ_SCALE_MAX = 0.41
    JITTERING = 0.01

    env_strength = np.random.uniform(0.7, 0.9)

    object_scales = [np.random.uniform(OBJ_SCALE_MIN, OBJ_SCALE_MAX)]*(1)

    plane_obj = render_utils.add_plane((0,0,0), 10.0)
    
    bg_pbr_path = json_data.get('bg_pbr_path')
    if bg_pbr_path is not None and bg_pbr_path != '':
        add_pbr_material(plane_obj.name, "floor_material", bg_pbr_path, scale = (10.0, 10.0, 10.0))

    env_hdr_path = json_data.get('env_hdr_path')
    if env_hdr_path is not None and env_hdr_path != '':
        render_utils.add_IBL(
                os.path.basename(env_hdr_path),
                os.path.dirname(env_hdr_path),
                env_strength
                )
   
    scn = bpy.context.scene
    
    load_end_time = load_start_time = time.time()


    bpy.context.view_layer.objects.active = None

    for o in bpy.data.objects:
        o.select_set(False)
    
    load_start_time = time.time()
    
    obj, bbox = render_utils.load_obj_v2(scn, json_data)

    vertices = np.array([v.co for v in obj.data.vertices])
    obj.scale = obj.scale * 0.5 / np.max(np.abs(vertices))
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)
    
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.01)
    bpy.ops.object.mode_set(mode='OBJECT')
    load_end_time = time.time()
    obj.select_set(True)  
    bpy.context.view_layer.objects.active = obj
    
    bpy.ops.mesh.customdata_custom_splitnormals_clear()

    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()
    
    
    ### rescaling object to actual desired scale
    obj.scale = [object_scales[0]]*3
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    ### calculate z_shift to put object on plane surface
    pose_mat = np.eye(4,4)
    
    vertices = np.array([v.co for v in obj.data.vertices])
    z_shift = np.abs(vertices[:,2].min())

    ### set location in the pose matrix
    pose_mat[0,-1] = 0
    pose_mat[1,-1] = 0
    pose_mat[2,-1] = z_shift
    
    obj.matrix_world = Matrix(pose_mat)

    bpy.context.view_layer.objects.active = obj

    (min_corner, max_corner), _ = get_bbox_of_the_scene(obj, attribute_name = "camera_target_ids")
    _, verts_indices = get_bbox_of_the_scene(obj, attribute_name = "visible_target_ids")

    bbox_center = (min_corner + max_corner) / 2
    bbox_size = max_corner - min_corner
    bbox_radius = bbox_size.length / 2  # half the diagonal

    # Draw the 3D bounding box
    #draw_bounding_box(*get_bbox_of_the_scene(obj)[0], name="PartID_1_BoundingBox", color=(0.0, 1.0, 0.0, 1.0))

    # Choose camera track-to point
    track_to_points_base = np.zeros((1, 3))
    track_to_points_base[0,:] = bbox_center

    # Update sample_camera_positions call with parameters from JSON
    sample_camera_positions_kwargs = {
        "azimuth_range": AZIMUTH_RANGE,
        "elevation_range": ELEVATION_RANGE,
        "distance_multiplier": DISTANCE_MULTIPLIER,
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
        
        rendered_camera_data = []
        
        #CAMERA_SEARCH_SPACE = 50
        initial_camera_positions = sample_camera_positions(
            bbox_center, bbox_radius, n_views=CAMERA_SEARCH_SPACE, **sample_camera_positions_kwargs
        )

        if VISIBILTY_CHECK_ENABLED:
            candidate_cameras = []
            for i, current_cam_position in enumerate(initial_camera_positions):
                # Set camera position and track-to point for evaluation
                parent_object.location = current_cam_position
                current_track_to_point = track_to_points_base[0] + np.random.uniform(-JITTERING, JITTERING, 3)
                constraint_object.location = current_track_to_point
                bpy.context.view_layer.update()

                visibilty_score = compute_visibility_score(cam, obj, verts_indices)
                # 2. Obtain visibility score for each camera. If is_inside is True, assign visibility score to zero.
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
                    "K": render_utils.get_calibration_matrix_K_from_blender(cam.data),
                    "RT": render_utils.get_3x4_RT_matrix_from_blender(cam),
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
            if len(initial_camera_positions) > N_FRAMES:
                initial_camera_positions = random.sample(initial_camera_positions, N_FRAMES)

            for i, current_cam_position in enumerate(initial_camera_positions):

                if is_inside(current_cam_position, obj):
                    pass
                else:
                    parent_object.location = current_cam_position
                    current_track_to_point = track_to_points_base[0] + np.random.uniform(-JITTERING, JITTERING, 3)
                    constraint_object.location = current_track_to_point
                    bpy.context.view_layer.update() # Update scene to get correct K and RT
                    
                    cameras_to_render.append({
                        "position": current_cam_position,
                        "track_to_point": current_track_to_point,
                        "visibility_score": 1.0, # Assign a dummy score as no check is performed
                        "K": render_utils.get_calibration_matrix_K_from_blender(cam.data),
                        "RT": render_utils.get_3x4_RT_matrix_from_blender(cam),
                    })
            
            # Respect N_FRAMES even when visibility check is disabled
            if len(cameras_to_render) > N_FRAMES:
                cameras_to_render = random.sample(cameras_to_render, N_FRAMES)
            
            print(f"Selected {len(cameras_to_render)} cameras for rendering (visibility check disabled).")

        # --- Hide vertices based on per_vertex_visibility attribute ---
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
            parent_object.keyframe_insert(data_path='location', frame=rendered_frames_count)
            
            constraint_object.location = cam_data["track_to_point"]
            constraint_object.keyframe_insert(data_path='location', frame=rendered_frames_count) # Keyframe the target as well
            
            rendered_camera_data.append({
                "frame": rendered_frames_count,
                "K": np.array(cam_data["K"]).tolist(),
                "RT": np.array(cam_data["RT"]).tolist(),
                "visibility_score": cam_data["visibility_score"],
            })
            
            bpy.ops.render.render(write_still=True)
            rendered_frames_count += 1
            print(f"Rendered frame {rendered_frames_count}: Visibility Score = {cam_data['visibility_score']:.4f}")

        scn.frame_end = rendered_frames_count - 1 if rendered_frames_count > 0 else 0 # Adjust scene frame end to actual rendered frames

        # Save camera information to a JSON file
        camera_info_filepath = os.path.join(output_dirpath, "camera_info.json")
        with open(camera_info_filepath, 'w') as f:
            json.dump(rendered_camera_data, f, indent=4)
        print(f"Saved camera information to {camera_info_filepath}")

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        #bpy.ops.wm.save_as_mainfile(filepath="/disk/nfs/gazinasvolume2/s2514643/sketch-assembly-AD/output_folder/scene.blend")


    except Exception as e:
        traceback.print_exc()
        print(f'err:  Exception: {e}')


if __name__ == "__main__":
    main()
