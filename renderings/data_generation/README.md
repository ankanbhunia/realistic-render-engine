# Demo
## Download Data

Download `assets.tar` from [here](https://www.dropbox.com/s/ah0obh8an0jgdr8/assets.tar?dl=0), untar the folder in `common`

Download `poses.tar` from [here](https://www.dropbox.com/s/mkrfnfm9vz5d6i8/poses.tar?dl=0), untar the folder in `common`

Move the sub-folders in `poses` to `common`. Inside `common`:
```
mv poses/* .
```
Download `toys_scene_configs.tar` from [here](https://www.dropbox.com/s/kbdib7kmcbihao9/toys_scene_configs.tar?dl=0), untar the folder in `common`

## Download Blender 2.93 LTS

```
https://www.blender.org/download/releases/2-93/
```
Adjust Toys4k data path in `rendering/render_utils.py`

After adjusting the paths in `rendering/render_toys.sh` run the following command inside `rendering` to make sure everything is setup and things are working properly
```
bash render_toys.sh
```

# Codebase details
## 1. Scene config generation
The code for this is located in `scene_config_generation`. We want to generate a `.json` file for each scene, picking the objects, their pose, the lighting environment and floor, and the camera pose all ahead of time before rendering. A sample script to do this is `TOYS_create_scene_configs.py`.
```
python TOYS_create_scene_configs.py
```
## 2. Rendering
Run `render_toys.sh` inside `rendering` with the appropriate paths. 
```
bash render_toys.sh
```
