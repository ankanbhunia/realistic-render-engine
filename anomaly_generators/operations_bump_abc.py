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


    #obj_paths_all = glob.glob('/home/s2514643/MOAD-data-generation-latest/abc_dataset/*.obj')
    base_path = '/home/s2514643/MOAD-data-generation-latest/abc_dataset/'
    out_folder = '/home/s2514643/MOAD-data-generation-latest/abc_anomaly/'
    
    existing_objs = glob.glob(out_folder+'/*')

    #for existing_obj in tqdm.tqdm(existing_objs):

    #    if len(os.listdir(existing_obj))>0:
    for i in tqdm.tqdm(range(1000)):

            existing_obj = np.random.choice(existing_objs)         
            obj_path = base_path+'/'+os.path.basename(existing_obj)+'.obj'
            model_name = os.path.basename(obj_path)[:-4]
            model_path = out_folder+'/'+model_name

            os.makedirs(model_path, exist_ok = True)

            HookDeformAnomalyGenerator(obj_path, model_path, model_name)
        