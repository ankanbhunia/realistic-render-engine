# Load dependencies
import anomaly_generators.fracture_modes.fracture_utility as fracture
import sys, bpy
import os, shutil
import tempfile, time
import glob, trimesh
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import multiprocessing

## GOAL: mesh_path -> [frac_path1, frac_path2, ...]

filename = "/disk/nfs/gazinasvolume2/s2514643/MOAD/input.obj"

def apply_remesh_modifier_trimesh(input_filepath):

    # Convert trimesh to Blender mesh
    bpy.ops.import_scene.obj(filepath=input_filepath)  # Replace with the actual path or import your trimesh in another way
    mesh_obj = bpy.data.objects[0]

    # Apply Remesh modifier
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.modifier_add(type='REMESH')
    remesh_modifier = mesh_obj.modifiers["Remesh"]

    # Set Remesh modifier parameters
    remesh_modifier.voxel_size = 0.1
    remesh_modifier.mode = 'VOXEL'

    # Apply the modifier
    bpy.ops.object.modifier_apply({"object": mesh_obj}, modifier="Remesh")

    # Convert the modified Blender mesh back to trimesh
    bpy.ops.export_scene.obj(filepath=input_filepath)  # Replace with the actual path or export your trimesh in another way

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    return True

def _align_meshes_with_optim(mesh1, mesh2, translation = None): # we translate mesh2 to align with mesh1

    def chamfer_distance(translation, mesh1, mesh2):
        translated_mesh1 = mesh1.copy()
        translated_mesh1.vertices += translation
        distance_matrix = cdist(translated_mesh1.vertices, mesh2.vertices)
        chamfer_dist = np.sum(np.min(distance_matrix, axis=1)) + np.sum(np.min(distance_matrix, axis=0))

        return chamfer_dist

    if translation is None:
        initial_translation = np.array([0.0, 0.0, 0.0])
        start_time = time.time()
        result = minimize(chamfer_distance, initial_translation, args=(mesh1.simplify_quadric_decimation(face_count = 5000), mesh2.simplify_quadric_decimation(face_count = 5000)), method='Nelder-Mead')
        print (f"log: {time.time()-start_time} time taken to post-process fractured part")
        translation = -result.x
        mesh2.apply_translation(translation)
    else:
        mesh2.apply_translation(translation)

    return mesh2, translation


def create_facture_anomaly(filename, temp_folder, do_post_processing = True):

    try:
        fracture.generate_fractures(filename, num_modes = 10, output_dir=temp_folder,verbose=True, \
            num_impacts=5, compressed=False,cage_size=5000,volume_constraint=1/20)
    except:
        print('log: error when creating fractures.')
        return [], []
        
    fractured_paths = glob.glob(temp_folder+'/fracture*')

    #### choose main fractured_breaks

    good_fractures = []
    good_fractures_other_halfs = []
    min_break, max_break = 0.05, 0.7
    for f_path in fractured_paths:
        trimesh_objs = [trimesh.load(_) for _ in glob.glob(f_path+'/*')]
        n_verts =  [obj.vertices.shape[0] for obj in trimesh_objs]
        for n_vert_ix, obj_ix in zip(n_verts, trimesh_objs):
            break_percentage = 1-n_vert_ix/sum(n_verts)
            if break_percentage>min_break and break_percentage<max_break:
                good_fractures.append(obj_ix)
                other_half = trimesh.util.concatenate(list(set(trimesh_objs) - {obj_ix}))
                good_fractures_other_halfs.append(other_half)

    if not do_post_processing:
        return [good_fractures, good_fractures_other_halfs]

    ### apply post-processing of good_fractures

    aligned_new_fracture = []
    input_mesh = trimesh.load(filename)
    transformd_input_mesh = trimesh.load(glob.glob(temp_folder+'/mode_0/*.obj')[0])
    scale_factor = np.max(np.abs(input_mesh.vertices-input_mesh.centroid))/np.max(np.abs(transformd_input_mesh.vertices-transformd_input_mesh.centroid))
    translation = None
    for frac in good_fractures:
        frac_copy = frac.copy()
        frac_copy.apply_scale(scale_factor)
        frac_copy, translation = _align_meshes_with_optim(input_mesh, frac_copy, translation)
        aligned_new_fracture.append(frac_copy)


    
    ### apply post-processing of good_fractures_other_halfs

    aligned_new_fracture_other_halfs = []
    for frac in good_fractures_other_halfs:
        frac_copy = frac.copy()
        frac_copy.apply_scale(scale_factor)
        frac_copy, translation = _align_meshes_with_optim(input_mesh, frac_copy, translation)
        aligned_new_fracture_other_halfs.append(frac_copy)
        

    #shutil.rmtree(temp_folder)

    temp_folder+'/output'

    for idx, (mesh1, mesh2) in enumerate(zip(aligned_new_fracture, aligned_new_fracture_other_halfs)):

        folder = f'{temp_folder}/output{idx}'
        os.makedirs(folder, exist_ok=True)
        mesh1.export(folder+'/mesh1.obj')
        mesh2.export(folder+'/mesh2.obj')

        apply_remesh_modifier_trimesh(folder+'/mesh1.obj')
        apply_remesh_modifier_trimesh(folder+'/mesh2.obj')

    print (f'log: total {len(aligned_new_fracture)} eligible fractured objs.')

    #result_queue.put([aligned_new_fracture, aligned_new_fracture_other_halfs])
    #return aligned_new_fracture, aligned_new_fracture_other_halfs
        

def start_thread(filename):
    temp_folder = tempfile.mkdtemp()
    process = multiprocessing.Process(target=create_facture_anomaly, args=(filename,temp_folder))
    process.start()
    process.join()

    folders = glob.glob(temp_folder+'/output*')
    aligned_new_fracture, aligned_new_fracture_other_halfs = [], []

    for folder in folders:
        aligned_new_fracture.append(trimesh.load(folder+'/mesh1.obj'))
        aligned_new_fracture_other_halfs.append(trimesh.load(folder+'/mesh2.obj'))

    shutil.rmtree(temp_folder)

    return aligned_new_fracture, aligned_new_fracture_other_halfs

