from anomaly_generators.utils import *
import anomaly_generators.fracture_modes.main as fcm
import tempfile, random
import glob, os, math
from gpytoolbox.copyleft import lazy_cage
from pymeshfix._meshfix import PyTMesh
from mathutils import Euler
import bmesh
from bpy import context

def FractureAnomalyGenerator(blend_file_path, output_folder, model_name, \
    min_region = 0.02, max_region = 1.0, modes = ['frac', 'crack', 'disc'], num_iters = 500):
    
    load_blend_file(blend_file_path)

    make_triangular_mesh()
    seperate_objects_based_on_materials()
    all_mesh_splits, all_obj_names, all_mesh_data = get_mesh_data()

    splits = all_mesh_splits
    whole_mesh = trimesh.util.concatenate(splits)
    whole_bbox = whole_mesh.bounding_box.bounds
    total_area = sum(whole_mesh.bounding_box.area_faces)

    mesh_part_segment_output = get_random_mesh_parts(splits, all_obj_names, min_region, max_region, num_iters, \
                    total_area, connection_check = 'exclude')

    if len(mesh_part_segment_output) == 0:
        mesh_part_segment_output = [splits, [], all_obj_names, [], 1]
        
    [excludes, includes, excludes_names, includes_names, exclude_area] = mesh_part_segment_output

    scene = bpy.context.scene
    
    join_all_objects()
    obj_to_cut = bpy.data.objects.get("object")

    bpy.context.collection.objects.unlink(obj_to_cut)

    query_mesh = trimesh.util.concatenate(excludes)
    main_mesh = trimesh.util.concatenate(includes)

    query_mesh = get_watertight_mesh(query_mesh, form_cage = False)

    if query_mesh.faces.shape[0] > 5000:
        query_mesh = query_mesh.simplify_quadric_decimation(5000)

    print (f'Verts: {query_mesh.vertices.shape[0]} Faces: {query_mesh.faces.shape[0]}') 

    temp_file_name = tempfile.NamedTemporaryFile().name
    query_mesh_save_path = temp_file_name + '.obj'
    query_mesh.export(query_mesh_save_path)

    fractured_meshes, fractured_meshes_oh = fcm.start_thread(query_mesh_save_path) ## very long job

    def _check_if_fracture_okay():
        mesh = load_scene_as_obj()
        return check_connected_component_fine(mesh) and get_connected_components(mesh) == 1

    def _clear_scene():
        temp_obj = bpy.data.objects.get('DIFFERENCE')
        if temp_obj:
            bpy.data.objects.remove(temp_obj, do_unlink=True)

        temp_obj = bpy.data.objects.get('INTERSECT')
        if temp_obj:
            bpy.data.objects.remove(temp_obj, do_unlink=True)

    out_path_logs = []

    for idx, (fr_mesh, fr_mesh_oh) in enumerate(zip(fractured_meshes, fractured_meshes_oh)):

        mode = random.choice(modes)

        if mode == 'frac':

            boolean_difference(obj_to_cut, fr_mesh_oh, ops = 'DIFFERENCE')
            point_cloud_label, mesh_label = create_point_label_segments(load_object_as_obj(bpy.data.objects['DIFFERENCE']), \
                fr_mesh_oh)

    
        if mode =='crack':
            

            boolean_difference(obj_to_cut, fr_mesh_oh, ops = 'DIFFERENCE')
            boolean_difference(obj_to_cut, fr_mesh_oh, ops = 'INTERSECT')

            translation_vector = [np.random.choice([-1,1])*i for i in np.random.uniform(0.02, 0.04, size=3)]
            rotation_matrix_z = rotation_matrix_4x4(np.array([0, 0, 1]), np.radians(np.random.choice([-1,1])*np.random.uniform(7,10)))
            rotation_matrix_z[:3,3] = translation_vector
            bpy.data.objects['INTERSECT'].matrix_world = rotation_matrix_z.T

            point_cloud_label, mesh_label = create_point_label_segments(load_object_as_obj(bpy.data.objects['DIFFERENCE']), \
                load_object_as_obj(bpy.data.objects['INTERSECT']))


        if mode =='disc':

            boolean_difference(obj_to_cut, fr_mesh_oh, ops = 'DIFFERENCE')
            boolean_difference(obj_to_cut, fr_mesh_oh, ops = 'INTERSECT')
            
            rand_material = bpy.data.materials.new(name="RandomMaterial")
            hsv = random_hsv_color()
            rgb_color = hsv_to_rgb(*hsv)
            rand_material.diffuse_color = (rgb_color[0],rgb_color[0],rgb_color[2], 1.0)
            rand_material.roughness = random.uniform(0.1, 1.0)

            bpy.data.objects['INTERSECT'].data.materials.clear()
            bpy.data.objects['INTERSECT'].data.materials.append(rand_material.copy())

            point_cloud_label, mesh_label = create_point_label_segments(load_object_as_obj(bpy.data.objects['DIFFERENCE']), \
                load_object_as_obj(bpy.data.objects['INTERSECT']))

        if not _check_if_fracture_okay():
            print ('log: fracture quality not good. ignored.')
            _clear_scene()
            continue

        out_blend_file_path = get_file_path(output_folder, model_name, mode)

        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend_file_path))
        point_cloud_label.export(out_blend_file_path.replace('.blend', '.ply'))
        mesh_label.export(out_blend_file_path.replace('.blend', '.obj'))
        out_path_logs.append(out_blend_file_path)

        _clear_scene()


    [os.remove(_) for _ in glob.glob(temp_file_name+'*')]

    return True if len(out_path_logs)>0 else False

def SimpleDeformAnoamlyGenerator(blend_file_path, output_folder, model_name, modes = ['bend', 'twist'], min_angle = 25, max_angle = 45):

    load_blend_file(blend_file_path)
    original_mesh = load_scene_as_obj()

    angle = math.radians(np.random.choice([-1,1])*np.random.uniform(30, 40)) 
    mode = np.random.choice(modes)

    if mode == 'bend': 
        deform_method = 'BEND'
    elif mode == 'twist':
        deform_method = 'TWIST'

    deform_axis = np.random.choice(['X','Y','Z'])

    obj = bpy.context.active_object

    bpy.ops.object.modifier_add(type='SIMPLE_DEFORM')
    modifier = obj.modifiers[-1]  # Get the last added modifier (Simple Deform)

    modifier.deform_method = deform_method  # You can change this to 'TWIST', 'TAPER', etc.
    modifier.angle = angle  # Set the angle for the deformation
    modifier.deform_axis = deform_axis 

    bpy.ops.object.modifier_apply({"object": obj}, modifier="SimpleDeform")

    final_anomaly_mesh = load_scene_as_obj()
    pntcloud_label = create_point_label(original_mesh, final_anomaly_mesh)
    
    out_blend_file_path = get_file_path(output_folder, model_name, mode)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend_file_path))
    pntcloud_label.export(out_blend_file_path.replace('.blend', '.ply'))
    final_anomaly_mesh.export(out_blend_file_path.replace('.blend', '.obj'))

    return True

def HookDeformAnomalyGenerator(blend_file_path, output_folder, model_name, mode = 'defr'):

    load_blend_file(blend_file_path)
    original_mesh = load_scene_as_obj()

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

    final_anomaly_mesh = load_scene_as_obj()

    pntcloud_label = create_point_label(original_mesh, final_anomaly_mesh)

    out_blend_file_path = get_file_path(output_folder, model_name, mode)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend_file_path))
    pntcloud_label.export(out_blend_file_path.replace('.blend', '.ply'))
    final_anomaly_mesh.export(out_blend_file_path.replace('.blend', '.obj'))


def BumpAnomalyGenerator(blend_file_path, output_folder, model_name, mode = 'bump'):

    load_blend_file(blend_file_path)

    original_mesh = load_scene_as_obj()
    obj = bpy.data.objects[0]

    obj_vertices = np.array([i.co for i in obj.data.vertices])
    max_length = (obj_vertices.max(0) - obj_vertices.min(0)).max()

    group = obj.vertex_groups.new(name = 'HookGroup')

    vertices_indices = [i.index for i in obj.data.vertices]
    group.add(vertices_indices, 1, 'REPLACE' )

    hook_object = bpy.data.objects.new("Hook", None)
    bpy.context.scene.collection.objects.link(hook_object)

    hook_modifier = obj.modifiers.new(name='Hook', type='HOOK')
    random_vert = random.choice([i for i in obj.data.vertices])
    surface_point = random_vert.co #random_point_on_surface(mesh) 
    normal_vector = surface_point/np.sqrt(np.sum(np.array(surface_point)**2))
    loc_vector = (np.array(list(normal_vector)+[1])@rotation_matrix_4x4([0,0,1], math.radians(30)))[:3]

    hook_modifier.object = hook_object
    hook_modifier.center = surface_point
    hook_modifier.vertex_group =  'HookGroup' 
    hook_modifier.falloff_radius = 2

    hook_object.location = random.choice([-1, 1]) * random.uniform(0.5, 0.7) * loc_vector 

    bpy.ops.object.modifier_apply({"object": obj}, modifier="Hook")

    bpy.data.objects.remove(hook_object, do_unlink=True)

    final_anomaly_mesh = load_scene_as_obj()

    pntcloud_label = create_point_label(original_mesh, final_anomaly_mesh)

    out_blend_file_path = get_file_path(output_folder, model_name, mode)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend_file_path))
    pntcloud_label.export(out_blend_file_path.replace('.blend', '.ply'))
    final_anomaly_mesh.export(out_blend_file_path.replace('.blend', '.obj'))

def MissingAnomalyGenerator(blend_file_path, output_folder, model_name, mode = 'miss', min_region = 0.01, max_region=0.5, num_iters = 2000):

    num_res = 1

    load_blend_file(blend_file_path)
    make_triangular_mesh()
    seperate_objects_based_on_materials()

    all_mesh_splits, all_obj_names, all_mesh_data = get_mesh_data()
    if len(all_mesh_splits)==1:
        print ('log: the mesh has only one part.')
        return False

    splits = all_mesh_splits
    whole_mesh = trimesh.util.concatenate(splits)
    whole_bbox = whole_mesh.bounding_box.bounds
    total_area = sum(whole_mesh.bounding_box.area_faces)
    out_list = []
    split_dims = [i.bounding_box.bounds for i in splits]
    
    print ('Searching for good candidates...')
    count = 0

    mesh_part_segment_output = get_random_mesh_parts(splits, all_obj_names, min_region, max_region, num_iters, \
    total_area, connection_check = 'exclude')

    print ('Searching Done...')

    if len(mesh_part_segment_output) == 0:
        return False

    [excludes, includes, excludes_names, includes_names, exclude_area] = mesh_part_segment_output

    exclude_mesh = trimesh.util.concatenate(excludes)
    include_mesh = trimesh.util.concatenate(includes)


    join_all_objects()

    obj = bpy.context.scene.objects[0]

    if seperate_portions_(obj.data, exclude_mesh):
        seperated_obj = bpy.context.scene.objects[-1]
        point_cloud_label, mesh_label = create_point_label_segments(load_object_as_obj(obj), \
                load_object_as_obj(seperated_obj))
        bpy.data.objects.remove(seperated_obj, do_unlink=True)

    out_blend_file_path = get_file_path(output_folder, model_name, mode)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend_file_path))
    
    point_cloud_label.export(out_blend_file_path.replace('.blend', '.ply'))
    mesh_label.export(out_blend_file_path.replace('.blend', '.obj'))

    return True

def Transform3DAnomalyGenerator(blend_file_path, output_folder, model_name, \
    min_region = 0.01, max_region = 0.5, num_iters = 2000,  \
       min_trans = 0.1, max_trans = 0.4, min_angle =  0.3, max_angle = 0.4, \
        min_scale = 0.7, max_scale = 0.8, max_try = 10, \
        source_material_path = None, \
        modes = ['tran', 'rota', 'scal', 'mats']):

    num_res = 1

    load_blend_file(blend_file_path)
    make_triangular_mesh()
    seperate_objects_based_on_materials()

    all_mesh_splits, all_obj_names, all_mesh_data = get_mesh_data()
    if len(all_mesh_splits)==1:
        print ('log: the mesh has only one part.')
        return False

    splits = all_mesh_splits
    whole_mesh = trimesh.util.concatenate(splits)
    whole_bbox = whole_mesh.bounding_box.bounds
    total_area = sum(whole_mesh.bounding_box.area_faces)
    out_list = []
    split_dims = [i.bounding_box.bounds for i in splits]
    
    print ('Searching for good candidates...')
    count = 0

    mesh_part_segment_output = get_random_mesh_parts(splits, all_obj_names, min_region, max_region, num_iters, \
    total_area, connection_check = 'exclude')

    if len(mesh_part_segment_output) == 0:
        return False

    [excludes, includes, excludes_names, includes_names, exclude_area] = mesh_part_segment_output
    mode = np.random.choice(modes)

    exclude_mesh = trimesh.util.concatenate(excludes)
    include_mesh = trimesh.util.concatenate(includes)

    join_all_objects()

    obj = bpy.context.scene.objects[0]

    if seperate_portions_(obj.data, exclude_mesh):
        seperated_obj = bpy.context.scene.objects[-1]

        if mode in ['tran', 'rota', 'scal']:

            matrix = get_random_transformation_matrix(exclude_mesh, include_mesh, mode, min_trans, max_trans, \
            min_angle, max_angle, min_scale, max_scale, max_try)
            if not isinstance(matrix, np.ndarray):
                return False
            seperated_obj.matrix_world = matrix.T

        elif mode in ['mats']:

            if source_material_path is None:
                return False

            with context.blend_data.libraries.load(source_material_path, link=False) as (data_from, data_to):
                data_to.objects = data_from.objects
            source_object = bpy.data.objects[-1]
            material_slot = random.choice(list(source_object.material_slots))
            seperated_obj.data.materials[0] = material_slot.material
        
    out_blend_file_path = get_file_path(output_folder, model_name, mode)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend_file_path))
    point_cloud_label, mesh_label = create_point_label_segments(load_object_as_obj(obj), \
                load_object_as_obj(seperated_obj))
    point_cloud_label.export(out_blend_file_path.replace('.blend', '.ply'))
    mesh_label.export(out_blend_file_path.replace('.blend', '.obj'))

    return True



if __name__ == "__main__":

    blend_file_path = '/home/s2514643/MOAD-data-generation-latest/toys4k_blend_files/bunny/bunny_008/bunny_008.blend'
    new_blend_file_path = 'modified.blend'
    #blend_file_path = random.choice(glob.glob('/disk/nfs/gazinasvolume1/datasets/toys4k_blend_files/mug/*/*'))
    MissingAnomalyGenerator(blend_file_path, new_blend_file_path)




# # min_region, max_region = 0.02, 1.0
# # blend_file_path = random.choice(glob.glob('/disk/nfs/gazinasvolume1/datasets/toys4k_blend_files/mug/*/*'))
# # #FractureAnomalyGenerator(blend_file_path, new_blend_file_path, min_region, max_region)
# # FractureAnomalyGenerator(blend_file_path, new_blend_file_path)

# blend_file_path = '/disk/nfs/gazinasvolume2/s2514643/MOAD-v1/submarine_011.blend'
# source_material_path = '/disk/nfs/gazinasvolume2/s2514643/MOAD-v1/submarine_004.blend'
# MaterialAnomalyGenerator(blend_file_path, new_blend_file_path, source_material_path)


# FractureAnomalyGenerator 
# SimpleDeformAnoamlyGenerator 
# HookDeformAnomalyGenerator 
# BumpAnomalyGenerator 
# MissingAnomalyGenerator 
# Transform3DAnomalyGenerator 
# MaterialAnomalyGenerator

# import scipy
# input_mesh = trimesh.load('/disk/nfs/gazinasvolume2/s2514643/MOAD-v1/file2.obj', force = 'mesh')
# deform_mesh = trimesh.load('/disk/nfs/gazinasvolume2/s2514643/MOAD-v1/file.obj',  force = 'mesh')


