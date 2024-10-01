import sys
sys.path.append('/home/s2514643/MOAD-data-generation-latest/MOAD-v1')
from anomaly_generators.utils import *
import anomaly_generators.fracture_modes.main as fcm
import tempfile, random
import glob, os, math
from gpytoolbox.copyleft import lazy_cage
from pymeshfix._meshfix import PyTMesh
from mathutils import Euler
import bmesh
from bpy import context
import concurrent.futures
import tqdm, shutil

def FractureAnomalyGenerator(obj_path, out_folder, model_name, modes = ['frac', 'crack']):

    temp_folder = tempfile.mkdtemp()

    [frac_mesh, frac_mesh_oh] = fcm.create_facture_anomaly(obj_path, temp_folder, do_post_processing = False)
    
    for idx, (fr_mesh, fr_mesh_oh) in enumerate(zip(frac_mesh, frac_mesh_oh)):

        mode = random.choice(modes)

        #model_name = os.path.basename(obj_path)[:-4]
        #os.makedirs(out_folder+'/'+model_name, exist_ok = True)

        output_file_path = get_file_path(out_folder, model_name, mode).replace('.blend', '.obj')

        if mode == 'frac':

            out_mesh = fr_mesh

        if mode == 'crack':

            translation_vector = [np.random.choice([-1,1])*i for i in np.random.uniform(0.02, 0.04, size=3)]
            rotation_matrix_z = rotation_matrix_4x4(np.array([0, 0, 1]), np.radians(np.random.choice([-1,1])*np.random.uniform(7,10)))
            rotation_matrix_z[:3,3] = translation_vector
            fr_mesh_oh.apply_transform(rotation_matrix_z)

            out_mesh = trimesh.util.concatenate([fr_mesh_oh, fr_mesh])

        out_mesh.export(output_file_path)

import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")

def FractureAnomalyGenerator_t(timeout_seconds, *args):
    # Set up the signal handler for SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)
    # Set the alarm to go off after the specified timeout
    signal.alarm(timeout_seconds)
    
    try:
        result = FractureAnomalyGenerator(*args)
    except TimeoutError:
        print("Function timed out")
        # Handle the timeout here, e.g., return a default value or raise an exception
    else:
        # Cancel the alarm if the function completes before the timeout
        signal.alarm(0)
        return result


def HookDeformAnomalyGenerator(obj_path, out_folder, model_name, mode = 'bump'):

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    bpy.ops.import_scene.obj(filepath=obj_path)

    obj = bpy.data.objects[0]

    obj_vertices = np.array([i.co for i in obj.data.vertices])
    max_length = (obj_vertices.max(0) - obj_vertices.min(0)).max()

    group = obj.vertex_groups.new(name = 'HookGroup' )

    # obj_mesh = bpy.data.meshes[9]
    vertices_indices = [i.index for i in obj.data.vertices]
    group.add(vertices_indices, 1, 'REPLACE' )

    # Create an Empty object to act as the hook
    hook_object = bpy.data.objects.new("Hook", None)
    bpy.context.scene.collection.objects.link(hook_object)

    # Add the Hook modifier to the object
    hook_modifier = obj.modifiers.new(name='Hook', type='HOOK')

    # Set the Empty object as the hook target
    hook_modifier.object = hook_object
    hook_modifier.vertex_group =  'HookGroup' 
    hook_modifier.falloff_radius = max_length/1.3
    
    hook_object.location = np.array([random.uniform(-1, 1), \
        random.uniform(-1, 1), \
        random.uniform(-1, 1)])

    angle = math.radians(30)
    hook_object.rotation_euler = Euler((random.uniform(-angle,angle), random.uniform(-angle, angle), random.uniform(-angle, angle)), 'XYZ')
    
    
    bpy.ops.object.modifier_apply({"object": obj}, modifier="Hook")
    
    bpy.data.objects.remove(hook_object, do_unlink=True)

    output_file_path = get_file_path_obj(out_folder, model_name, mode).replace('.blend', '.obj')

    bpy.ops.export_scene.obj(filepath=output_file_path, use_selection=True)



if __name__ == "__main__":

    #obj_paths_all = glob.glob('/disk/scratch/s2514643/abc_dataset/*/*.obj')
    #out_folder = '/home/s2514643/MOAD-data-generation-latest/abc_dataset/'

    #for d in tqdm.tqdm(obj_paths_all):
    #    shutil.copy(d, os.path.join(out_folder, os.path.basename(d)))
    
    obj_paths_all = glob.glob('/home/s2514643/MOAD-data-generation-latest/abc_dataset/*.obj')
    out_folder = '/home/s2514643/MOAD-data-generation-latest/abc_anomaly/'

    while True:

        obj_path = np.random.choice(obj_paths_all)

        model_name = os.path.basename(obj_path)[:-4]
        model_path = out_folder+'/'+model_name
        
        if os.path.isdir(model_path):
            continue

        os.makedirs(model_path, exist_ok = True)

        FractureAnomalyGenerator_t(900, obj_path, model_path, model_name)