import os, sys, json
import subprocess, tempfile
from pathlib import Path
try:
    from .utils.geo_utils import simulate_to_get_stable_pose
except ImportError:
    from utils.geo_utils import simulate_to_get_stable_pose
import numpy as np


def render_segmentation_object_with_blender(
    input_json: str,
    output_path: str,
) -> None:

    abs_path = Path(__file__).resolve().parent

    # Ensure the output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Construct the command for the subprocess call
    command = [
        f"{abs_path}/blender-2.93.18-linux-x64/blender",
        "-b",  # Run in background mode (no GUI)
        "-P", f"{abs_path}/utils/render_wrapper_segmentation.py",  # Python script to execute
        "--",  # Separator for script arguments
        "--input_json", str(os.path.abspath(input_json)),
        "--output_path", str(os.path.abspath(output_path)),
    ]

    print("Executing Blender rendering command...")
    
    result = subprocess.run(command)#, capture_output=True, text=True, check=False)


def render_untextured_object_with_blender(
    input_json: str,
    output_path: str,
    mode: list[str] = ['lineart']
) -> None:
    """
    Renders an untextured 3D object from multiple camera angles using Blender.
    Launches a separate Blender process for each specified rendering mode to ensure isolation.

    Args:
        input_json (str): Path to the input JSON file containing object data.
        output_path (str): Base directory where rendered images will be saved.
        mode (list[str]): List of rendering modes. Can be "lineart", "textureless", "depth".
                          Default is ["lineart"].
    """
    abs_path = Path(__file__).resolve().parent

    # Ensure the base output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for single_mode in mode:
        # Create a mode-specific output directory to avoid conflicts
        # mode_output_path = Path(output_path) / single_mode
        # mode_output_path.mkdir(parents=True, exist_ok=True)

        command = [
            f"{abs_path}/blender-2.93.18-linux-x64/blender",
            "-b",  # Run in background mode (no GUI)
            f"{abs_path}/utils/empty_scene.blend",
            "-P", f"{abs_path}/utils/render_wrapper_untextured.py",  # Python script to execute
            "--",  # Separator for script arguments
            "--input_json", str(os.path.abspath(input_json)),
            "--out_path", str(os.path.abspath(output_path)), # Use mode-specific output path
            "--mode", single_mode # Pass a single mode string
        ]

        print(f"Executing Blender rendering command for mode '{single_mode}'...")
        
        result = subprocess.run(command)#, capture_output=True, text=True, check=False)

def render_photorealistic_object_with_blender(
    input_json: str,
    output_path: str,
):

    abs_path = Path(__file__).resolve().parent

    command = [
        f"{abs_path}/blender-2.93.18-linux-x64/blender",
        "--background",
        f"{abs_path}/utils/empty_scene.blend",
        "--python",
        f"{abs_path}/utils/render_wrapper_photorealistic.py",
        "--",
        "--input_json", os.path.abspath(input_json),
        "--output_path", os.path.abspath(output_path),
        # N_FRAMES, visibility_threshold, azimuth_range, elevation_range, and distance_multiplier
        # are now loaded directly from input_json within render_wrapper_photorealistic.py
    ]

    print (command)
    result = subprocess.run(command)#, capture_output=True, text=True, check=False)

def create_render_json(
    obj_list,
    visible_target_ids=None,
    camera_target_ids=None,
    global_pose=None,
    env_hdr_path=None,
    bg_pbr_path=None,
    num_views=30,
    camera_search_space=30,
    visibility_check_enabled=True,
    azimuth_range=(0, 360),
    elevation_range=(-30, 60),
    distance_multiplier=(1.5, 2.5),
    init_dict=None,
    seed: int = None
):
    """
    Constructs a JSON dictionary for rendering based on provided object data and scene parameters.

    Args:
        obj_list (list): A list of dictionaries, where each dictionary represents an object
                         with keys like 'obj_name', 'obj_path', 'mat_path', 'transform'.
        camera_focus_obj_ids (list, optional): List of object IDs to focus the camera on.
        global_pose (list): A list representing the global pose matrix.
        env_hdr_path (str): Path to the environment HDR file.
        bg_pbr_path (str): Path to the background PBR material.
        num_views (int): Total number of views to render.
        azimuth_range (tuple): (min_azimuth_deg, max_azimuth_deg) for camera sampling.
        elevation_range (tuple): (min_elevation_deg, max_elevation_deg) for camera sampling.
        distance_multiplier (tuple): (min_multiplier, max_multiplier) for camera distance.
        init_dict (dict, optional): A dictionary containing default parameters like
                                    'render_parameters' and 'camera_parameters'.
                                    If None, default values will be used.

    Returns:
        dict: The JSON structure for rendering.
    """
    # Set default init_dict if not provided
    if init_dict is None:
        init_dict = {
            "render_parameters": {
                "render_tile_x": 112,
                "render_tile_y": 112,
                "resolution_x": 1024,
                "resolution_y": 1024,
                "use_denoising": True,
                "use_persistent_data": True,
                "use_save_buffers": True,
                "samples": 256,
                "use_spatial_splits": True,
                "max_bounces": 10,
                "min_bounces": 2,
                "use_caustics_reflective": True,
                "use_caustics_refractive": True
            },
            "camera_parameters": {
                "focal_length": 30,
                "sensor_height": 36,
                "sensor_width": 36
            }
        }

    # Create a new dictionary to avoid modifying the original init_dict
    output_data = init_dict.copy()

    # Update the dictionary with the provided scene data
    output_data.update({
        "objects": obj_list,
        "visible_target_ids": visible_target_ids,
        "camera_target_ids": camera_target_ids,
        "global_pose": global_pose,
        "env_hdr_path": os.path.abspath(env_hdr_path) if env_hdr_path else None,
        "bg_pbr_path": os.path.abspath(bg_pbr_path) if bg_pbr_path else None,
        "num_views": num_views,
        "camera_search_space": camera_search_space,
        "visibility_check_enabled":visibility_check_enabled,
        "azimuth_range": azimuth_range,
        "elevation_range": elevation_range,
        "distance_multiplier": distance_multiplier,
    })
    if seed is not None:
        output_data["seed"] = seed

    return output_data


# if __name__ == "__main__":

