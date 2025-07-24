
try:
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    import cv2
    OPENEXR_CV2_AVAILABLE = True
except:
    OPENEXR_CV2_AVAILABLE = False
import bpy
import json
import math
import random
import mathutils
import sys, os
import argparse
from mathutils import Matrix, Vector
import numpy as np
import bpy_extras

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

def normalize_to_unit_cube(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bbox = [mathutils.Vector(corner) for corner in obj.bound_box]
    min_corner = mathutils.Vector(map(min, zip(*bbox)))
    max_corner = mathutils.Vector(map(max, zip(*bbox)))
    size = max_corner - min_corner
    scale = 1.0 / max(size)
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(scale=True)
    obj.location = (0, 0, 0)


def apply_transform_matrix(obj, matrix):
    obj.matrix_world = mathutils.Matrix(matrix)


def setup_camera(azimuth, elevation, object_size=1.0, margin=1.0):
    # Remove old cameras
    for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
        bpy.data.objects.remove(cam, do_unlink=True)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    # Set camera type to perspective
    cam_data.type = 'PERSP'
    cam_data.sensor_fit = 'AUTO'

    # Calculate distance to fit object in frame
    # Assuming object is centered at origin and scaled to unit cube (max dimension 1)
    # Field of View (FOV) calculation based on sensor size and focal length
    # For a unit cube, the diagonal is sqrt(3). To fit it, we need to consider the largest dimension.
    # Since frame_all_objects_in_unit_cube scales to max dimension 1, we can use 1.0 as object_size.
    
    # Get scene resolution
    render_res_x = bpy.context.scene.render.resolution_x
    render_res_y = bpy.context.scene.render.resolution_y
    
    # Determine which dimension (width or height) is the limiting factor for FOV
    # The FOV is typically defined by the sensor's larger dimension relative to the image aspect ratio
    aspect_ratio = render_res_x / render_res_y
    
    # Use the larger of sensor_width or sensor_height, adjusted by aspect ratio
    # to ensure the object fits in both dimensions.
    # For simplicity, let's assume we want to fit the object's largest dimension (1.0)
    # within the camera's view.
    
    # Calculate horizontal and vertical FOV
    # Blender's camera sensor width/height directly relates to FOV
    # FOV = 2 * atan(sensor_size / (2 * focal_length))
    
    # To fit a unit cube (max dimension 1) with a margin, the effective object size is 1.0 * margin
    effective_object_size = object_size * margin
    
    # Calculate distance based on the smaller of the two FOVs (horizontal or vertical)
    # to ensure the object fits completely.
    # For a perspective camera, the distance D to fit an object of size S is:
    # D = S / (2 * tan(FOV/2))
    
    # Let Blender calculate the FOV angles based on sensor size, focal length, and render aspect ratio
    bpy.context.view_layer.update()
    fov_x = cam_data.angle_x
    fov_y = cam_data.angle_y

    # Determine the required distance based on the object's dimensions and camera FOV
    # We need to consider both width and height of the object relative to the camera's FOV
    # Assuming the object is a unit cube, its "width" and "height" are 1.0
    
    # Distance needed to fit object horizontally
    dist_x = (effective_object_size / 2) / math.tan(fov_x / 2)
    # Distance needed to fit object vertically
    dist_y = (effective_object_size / 2) / math.tan(fov_y / 2)
    
    # The actual distance should be the maximum of these two to ensure the object fits in both dimensions
    dist = max(dist_x, dist_y)

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    x = dist * math.cos(el_rad) * math.cos(az_rad)
    y = dist * math.cos(el_rad) * math.sin(az_rad)
    z = dist * math.sin(el_rad)
    cam.location = (x, y, z)

    direction = mathutils.Vector((0, 0, 0)) - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = cam


import bpy
from mathutils import Vector

def frame_all_objects_in_unit_cube():
    """
    Scales and translates all mesh objects in the scene so that they fit
    within a unit cube centered at the world origin.
    """
    # --- 1. Find the combined bounding box of all mesh objects ---
    
    # Start with inverted min/max to ensure the first object's bounds are captured
    global_min = Vector((float('inf'), float('inf'), float('inf')))
    global_max = Vector((float('-inf'), float('-inf'), float('-inf')))

    # Get all mesh objects in the current scene
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    if not mesh_objects:
        print("No mesh objects found in the scene.")
        return

    # Iterate through each mesh object to find the overall bounds
    for obj in mesh_objects:
        # Get the bounding box corners in world space
        # obj.bound_box gives 8 corners in local space.
        # We transform each corner to world space to get the true bounding box.
        world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        
        # Update the global min/max with the corners of this object
        for corner in world_corners:
            global_min.x = min(global_min.x, corner.x)
            global_min.y = min(global_min.y, corner.y)
            global_min.z = min(global_min.z, corner.z)
            
            global_max.x = max(global_max.x, corner.x)
            global_max.y = max(global_max.y, corner.y)
            global_max.z = max(global_max.z, corner.z)

    # --- 2. Calculate the required scale and translation ---

    # Calculate the center of the bounding box
    center = (global_min + global_max) / 2.0

    # Calculate the size of the bounding box
    size = global_max - global_min

    # Find the largest dimension of the bounding box
    if max(size) == 0:
        print("Cannot scale an object with zero size.")
        return
        
    scale_factor = 1.0 / max(size)

    # --- 3. Apply the transformation to all mesh objects ---

    print(f"Scene center: {center}")
    print(f"Scene size: {size}")
    print(f"Calculated scale factor: {scale_factor}")

    # Iterate through all mesh objects and apply the transformation
    for obj in mesh_objects:
        # First, translate the object's origin to the scene's center
        obj.location -= center
        
        # Then, scale the object and its location
        obj.location *= scale_factor
        obj.scale *= scale_factor

    print("\nAll mesh objects have been framed within a unit cube.")

def import_objects(data_list):
    for obj_data in data_list:
        path = obj_data['obj_path']
        matrix = obj_data['transform']
        label = obj_data['obj_seg_label']

        bpy.ops.import_scene.obj(filepath=path)
        imported_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        for obj in imported_objs:
            # normalize_to_unit_cube(obj)
            obj.location = (0, 0, 0)
            mesh = obj.data
            
            # Transform each vertex
            for vertex in mesh.vertices:
                # Create a 4D homogeneous vector (w=1)
                v_hom = vertex.co.to_4d()
                # Apply the transformation
                v_transformed_hom = Matrix(matrix) @ v_hom
                # Convert back to 3D (handles perspective division)
                vertex.co = v_transformed_hom.to_3d()
                
            mesh.update()
            
            obj.pass_index = label
    frame_all_objects_in_unit_cube()


def setup_render_common(x_dimension, y_dimension):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.render.resolution_x = x_dimension
    scene.render.resolution_y = y_dimension
    scene.view_layers["View Layer"].use_pass_object_index = True


def render_rgb(output_path):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def render_segmentation(output_path):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["IndexOB"], comp.inputs["Image"])

    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.image_settings.color_depth = '32'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def get_visible_segmentation_labels(image_path):
    if not OPENEXR_CV2_AVAILABLE:
        return set()

    exr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if exr is None:
        print(f"Failed to read EXR file at {image_path}")
        return set()

    if exr.ndim == 3:
        seg_mask = exr[:, :, 0]
    else:
        seg_mask = exr
    
    unique_labels = np.unique(seg_mask)
    # Filter out the background label (0)
    visible_labels = set(unique_labels[unique_labels > 0].astype(int))
    return visible_labels


def compute_visibility_score(cam_obj, obj_to_check, vertices_to_check):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj_to_check.evaluated_get(depsgraph)
    matrix_world = obj_to_check.matrix_world
    cam_location = cam_obj.matrix_world.translation
    visible_vertices = []
    total_vertices = len(vertices_to_check)

    if total_vertices == 0:
        return 0.0

    for v_index in vertices_to_check:
        vert_co_world = matrix_world @ obj_to_check.data.vertices[v_index].co
        co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, vert_co_world)
        
        if (0.0 < co_ndc.x < 1.0 and 0.0 < co_ndc.y < 1.0 and co_ndc.z > 0):
            direction = (vert_co_world - cam_location).normalized()
            hit, location, normal, index, hit_obj, matrix = scene.ray_cast(depsgraph, cam_location, direction)
            
            if hit:
                distance_to_hit = (location - cam_location).length
                distance_to_vert = (vert_co_world - cam_location).length
                
                if abs(distance_to_hit - distance_to_vert) < 0.001:
                    visible_vertices.append(v_index)
            else:
                visible_vertices.append(v_index)
            
    score = len(visible_vertices) / total_vertices
    return score


def save_segmentation_as_colored_image(exr_path, output_path, all_labels_in_scene, seed=None):
    if not OPENEXR_CV2_AVAILABLE:
        print("OpenCV or OpenEXR not available, cannot save colored segmentation.")
        return

    if seed is not None:
        # Use a local random state for color generation to not interfere with global seed for camera placement
        local_random = random.Random(seed)
    else:
        local_random = random

    exr = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if exr is None:
        print(f"Failed to read EXR file at {exr_path}")
        return

    if exr.ndim == 3:
        seg_mask = exr[:, :, 0]
    else:
        seg_mask = exr
    
    seg_mask = seg_mask.astype(np.int32)
    
    # Create a persistent color map for all labels in the scene for color consistency
    # across different views.
    if not hasattr(save_segmentation_as_colored_image, "color_map"):
        save_segmentation_as_colored_image.color_map = {0: (0, 0, 0)} # Background is black
        # Sort labels to ensure deterministic color assignment if script is re-run
        for label in sorted(list(all_labels_in_scene)):
            if label not in save_segmentation_as_colored_image.color_map:
                # Generate a random color using the local random state
                save_segmentation_as_colored_image.color_map[label] = (local_random.randint(50, 255), local_random.randint(50, 255), local_random.randint(50, 255))

    color_map = save_segmentation_as_colored_image.color_map
    
    # Create an RGB image from the segmentation mask
    rgb_image = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    
    unique_labels_in_view = np.unique(seg_mask)
    
    for label in unique_labels_in_view:
        if label == 0:
            continue
        if label in color_map:
            rgb_image[seg_mask == label] = color_map[label]
        else:
            # Assign a new random color if a label appears that was not in the initial set
            # This is a fallback, ideally all_labels_in_scene should be complete.
            new_color = (local_random.randint(50, 255), local_random.randint(50, 255), local_random.randint(50, 255))
            color_map[label] = new_color
            rgb_image[seg_mask == label] = new_color

    cv2.imwrite(output_path, rgb_image)
    print(f"Saved colored segmentation map to {output_path}")


def main(json_path, output_path):
    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with open(json_path, 'r') as f:
        data = json.load(f)

    seed = data.get("seed", None) 
    set_random_seed(seed)
    
    image_width, image_height = data["render_parameters"]["resolution_x"], data["render_parameters"]["resolution_y"] 
    
    azimuth_range = data.get("azimuth_range", [0, 360])
    elevation_range = data.get("elevation_range", [20, 60])


    setup_render_common(image_width, image_height)
    import_objects(data['objects'])
    scene = bpy.context.scene

    ### add point lights
    for rid in [270,90,0,180]:
        key_light = bpy.data.lights.new(name="Key Light", type='POINT')
        key_light.energy = 666  # Adjust the strength of the light
        key_light_obj = bpy.data.objects.new(name="KeyLight", object_data=key_light)
        scene.collection.objects.link(key_light_obj)
        key_light_obj.location = (5*math.cos(math.radians(rid)), 5*math.sin(math.radians(rid)), 5)   # Adjust the position of the light

    # Get all unique segmentation labels present in the scene
    all_labels = set(obj.pass_index for obj in bpy.context.scene.objects if obj.type == 'MESH' and obj.pass_index > 0)
    if not all_labels:
        print("No objects with segmentation labels found in the scene. Exiting.")
        return

    print(f"Total unique segmentation labels to capture: {all_labels}")

    # --- Iterative, Visibility-based View Selection ---
    
    VERTEX_SAMPLE_SIZE = 100    # Number of vertices to sample per object for speed
    CAMERA_SEARCH_SAMPLES = 64  # How many camera positions to check in a grid
    VISIBILITY_THRESHOLD = 0.1
    MAX_VIEWS = 12 # Safety break to prevent infinite loops
    min_views = 3 # At least how many views need to render

    # Group objects by their segmentation label
    objects_by_label = {}
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.pass_index > 0:
            label = obj.pass_index
            if label not in objects_by_label:
                objects_by_label[label] = []
            objects_by_label[label].append(obj)

    # Pre-sample vertices for each object to speed up checks
    sampled_vertices_by_obj = {}
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.pass_index > 0:
            all_verts = list(range(len(obj.data.vertices)))
            if len(all_verts) > VERTEX_SAMPLE_SIZE:
                sampled_vertices_by_obj[obj.name] = random.sample(all_verts, VERTEX_SAMPLE_SIZE)
            else:
                sampled_vertices_by_obj[obj.name] = all_verts

    # Generate random camera positions
    camera_positions = []
    if CAMERA_SEARCH_SAMPLES > 0:
        for _ in range(CAMERA_SEARCH_SAMPLES):
            az = random.uniform(azimuth_range[0], azimuth_range[1])
            el = random.uniform(elevation_range[0], elevation_range[1])
            camera_positions.append((az, el))
        print(f"Generated {len(camera_positions)} random camera positions.")

    MIN_ANGLE_DIFFERENCE = 30 # degrees, minimum angular difference for diverse views

    # Group objects by their segmentation label
    objects_by_label = {}
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.pass_index > 0:
            label = obj.pass_index
            if label not in objects_by_label:
                objects_by_label[label] = []
            objects_by_label[label].append(obj)

    # Pre-sample vertices for each object to speed up checks
    sampled_vertices_by_obj = {}
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.pass_index > 0:
            all_verts = list(range(len(obj.data.vertices)))
            if len(all_verts) > VERTEX_SAMPLE_SIZE:
                sampled_vertices_by_obj[obj.name] = random.sample(all_verts, VERTEX_SAMPLE_SIZE)
            else:
                sampled_vertices_by_obj[obj.name] = all_verts

    # Generate a pool of random camera positions.
    # Generate more samples than CAMERA_SEARCH_SAMPLES to increase chances of finding diverse views.
    num_candidate_positions = max(CAMERA_SEARCH_SAMPLES * 4, min_views * 10) # Ensure enough candidates
    camera_positions_pool = []
    for _ in range(num_candidate_positions):
        az = random.uniform(azimuth_range[0], azimuth_range[1])
        el = random.uniform(elevation_range[0], elevation_range[1])
        camera_positions_pool.append((az, el))
    print(f"Generated {len(camera_positions_pool)} random camera positions for the pool.")

    uncaptured_labels = all_labels.copy()
    rendered_views = [] # Store (az, el) tuples of rendered views
    view_count = 0

    while (uncaptured_labels or view_count < min_views) and view_count < MAX_VIEWS:
        print(f"\n--- Iteration {view_count + 1} ---")
        print(f"Labels remaining to be captured: {uncaptured_labels}")

        selected_az = None
        selected_el = None
        labels_captured_by_current_view = [] # Labels captured by the *selected* view in this iteration

        if view_count == 0:
            # First camera: completely random from the generated pool
            if not camera_positions_pool:
                print("Error: No camera positions available in the pool. Stopping.")
                break
            selected_az, selected_el = random.choice(camera_positions_pool)
            camera_positions_pool.remove((selected_az, selected_el)) # Remove to avoid re-selection
            print(f"  - Selecting first random view (Az: {selected_az:.2f}, El: {selected_el:.2f})")
        elif view_count < min_views:
            # For subsequent views up to min_views, ensure significant difference
            best_candidate = None
            max_min_dist = -1.0

            # Filter out already rendered views from the candidate pool
            available_candidates = [p for p in camera_positions_pool if p not in rendered_views]
            if not available_candidates:
                print("No more unique camera positions to check for diverse views. Stopping.")
                break

            for cand_az, cand_el in available_candidates:
                min_dist_to_rendered = float('inf')
                for prev_az, prev_el in rendered_views:
                    # Simple Euclidean distance in (az, el) space
                    dist = math.sqrt((cand_az - prev_az)**2 + (cand_el - prev_el)**2)
                    min_dist_to_rendered = min(min_dist_to_rendered, dist)
                
                if min_dist_to_rendered > max_min_dist:
                    max_min_dist = min_dist_to_rendered
                    best_candidate = (cand_az, cand_el)
            
            if best_candidate and max_min_dist >= MIN_ANGLE_DIFFERENCE:
                selected_az, selected_el = best_candidate
                camera_positions_pool.remove((selected_az, selected_el))
                print(f"  - Selecting diverse view for minimum views requirement (Az: {selected_az:.2f}, El: {selected_el:.2f}), min_dist: {max_min_dist:.2f}")
            else:
                # If no sufficiently diverse view found, and we still need min_views,
                # this is a critical failure as per the user's request.
                print(f"  - Error: Could not find a sufficiently diverse view (min_dist < {MIN_ANGLE_DIFFERENCE:.2f}) for minimum views requirement. Max min_dist found: {max_min_dist:.2f}. Stopping.")
                break # Cannot proceed if diversity requirement not met

        else: # view_count >= min_views
            # After min_views, revert to original logic: find view that captures new labels
            # Iterate through available positions to find one that captures new labels
            available_positions_for_labels = [p for p in camera_positions_pool if p not in rendered_views]
            if not available_positions_for_labels:
                print("No more camera positions to check for capturing new labels. Stopping.")
                break

            found_label_capturing_view = False
            for i, (az, el) in enumerate(available_positions_for_labels):
                print(f"  - Evaluating view {i+1}/{len(available_positions_for_labels)}...", end='\r')
                setup_camera(azimuth=az, elevation=el)
                cam = bpy.context.scene.camera
                
                current_view_captured_labels = []
                
                # Check visibility for only the uncaptured labels
                for label in uncaptured_labels:
                    max_score_for_label = 0
                    for obj in objects_by_label[label]:
                        if obj.name in sampled_vertices_by_obj:
                            score = compute_visibility_score(cam, obj, sampled_vertices_by_obj[obj.name])
                            if score > max_score_for_label:
                                max_score_for_label = score
                    
                    if max_score_for_label >= VISIBILITY_THRESHOLD:
                        current_view_captured_labels.append(label)
                
                if current_view_captured_labels:
                    # This view captures new labels, so we use it
                    selected_az, selected_el = az, el
                    labels_captured_by_current_view = current_view_captured_labels
                    found_label_capturing_view = True
                    camera_positions_pool.remove((selected_az, selected_el))
                    break # Break from the inner loop (available_positions_for_labels)
            
            if not found_label_capturing_view:
                print("Could not find a new view to capture any remaining labels. Stopping.")
                break

        # If a view was successfully selected in any of the above conditions
        if selected_az is not None and selected_el is not None:
            setup_camera(azimuth=selected_az, elevation=selected_el) # Ensure camera is set up for the selected view
            print(f"\nSelected view (Az: {selected_az:.2f}, El: {selected_el:.2f})")
            
            # Render and save
            seg_output_path = f"{output_path}/seg_map_{view_count:04d}.exr"
            render_segmentation(seg_output_path)
            colored_seg_path = f"{output_path}/seg_map_colored_{view_count:04d}.png"
            save_segmentation_as_colored_image(seg_output_path, colored_seg_path, all_labels, seed)
            render_rgb(f"{output_path}/mesh_{view_count:04d}.png")
            
            rendered_views.append((selected_az, selected_el))
            
            # Update uncaptured_labels only if we are past min_views or if labels were actually captured
            if view_count >= min_views or labels_captured_by_current_view:
                for label in labels_captured_by_current_view:
                    if label in uncaptured_labels:
                        uncaptured_labels.remove(label)
            
            print(f"Labels captured in this view: {labels_captured_by_current_view}")
            view_count += 1
        else:
            print("No view selected in this iteration. Stopping.")
            break

    print(f"\nFinished rendering. Total views rendered: {view_count}")
    if uncaptured_labels:
        print(f"Warning: Could not capture all labels. Remaining: {uncaptured_labels}")
    else:
        print("All labels successfully captured with sufficient visibility.")

    #bpy.ops.wm.save_as_mainfile(filepath="/disk/nfs/gazinasvolume2/s2514643/sketch-assembly-AD/output_folder/scene.blend")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D objects using Blender.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file containing object data.")
    parser.add_argument("--output_path", type=str, required=True, help="Path for the output image.")

    # This is a common pattern when running Blender scripts:
    # Blender passes its own arguments, and then '--' separates them from script arguments.
    # We need to parse only the arguments after '--'.
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv) # No '--' found, parse all arguments

    args = parser.parse_args(argv[index:])

    main(args.input_json, args.output_path)
