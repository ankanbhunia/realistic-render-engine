import bpy,math,time
import numpy as np
import trimesh, random, scipy
#import sklearn.cluster as skc
from pymeshfix._meshfix import PyTMesh
#import cc3d, 
import tempfile, os
from trimesh.voxel import creation
from scipy.spatial.transform import Rotation
from gpytoolbox.copyleft import lazy_cage
import bpy, mathutils, math, json
import random
import colorsys
#from sklearn.cluster import OPTICS
from mathutils import Vector

def random_hsv_color():
    h = random.random()  # Random hue
    s = random.uniform(0.1, 0.25)  # Random saturation
    v = random.uniform(0.3, 0.7)  # Random value
    return (h, s, v)

# Convert HSV color to RGB
def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)


def check_if_anomaly_visible(obj_path, ply_path, min_vis_points = 10, num_views_check = 8, height = 5, radius = 5):

    cam_pos_list = np.stack([radius*np.sin(np.radians(np.linspace(0,360,num_views_check))), \
        radius*np.cos(np.radians(np.linspace(0,360,num_views_check))), \
            np.array([height]*num_views_check)],1)[:-1]

    if isinstance(obj_path, str):
        mesh = trimesh.load(obj_path, force = 'mesh')
    else:
        mesh = obj_path

    if isinstance(ply_path, str):
        pnc = trimesh.load(ply_path)
    else:
        pnc = ply_path

    points = pnc.vertices
    colors = pnc.colors

    total_num_anomaly_points = sum((colors==255).all(1)==False)
    ray_tracer = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    out_list = []
    for camera_point in cam_pos_list:

        ray_origins = camera_point - points+points
        ray_directions = points - camera_point
        intersections,_,_ = ray_tracer.intersects_location(
                        ray_origins=ray_origins,
                        ray_directions=ray_directions,
                        multiple_hits = False
                    )
        is_visible = scipy.spatial.distance.cdist(points, intersections).min(1)<1e-5
        n_points_view = sum((colors[np.where(is_visible)]==255).all(1)==False)
        if n_points_view>min_vis_points:
            return True
        out_list.append (n_points_view)

        print (f'{n_points_view} - log-checking with different view-point.')
        
    return False
    

def simulate_to_get_stable_pose(blend_file_path, plane_height = 5, rot_degree = [0,0,0], num_frames = 250):

    if blend_file_path.endswith('.blend'):

        bpy.ops.wm.open_mainfile(filepath=blend_file_path)
        join_all_objects()

    elif blend_file_path.endswith('.obj'):

        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Set up a new scene
        scene = bpy.context.scene
        bpy.ops.import_scene.obj(filepath = blend_file_path, use_split_objects=False, use_split_groups=False)
        #join_all_objects()

        myobj = bpy.data.objects[0]
        myobj.name = "object"

        myobj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        myobj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        # Calculate the maximum dimension of the object
        max_dim = max(myobj.dimensions)
        # Calculate the scale factor to fit within the range 0 to 1
        scale_factor = 20 / max_dim
        # Scale the object
        myobj.scale *= scale_factor
        bpy.ops.object.transform_apply(scale=True)

    bpy.ops.mesh.primitive_plane_add(size=10)
    
    plane = bpy.context.object
    plane.scale = (3, 3, 1)  

    obj = bpy.data.objects["object"]#.get('object_0')

    min_obj_z = np.array([list(i.co) for i in obj.data.vertices])[:,2].min()
    plane.location.z = min_obj_z-plane_height

    rotation_matrix = mathutils.Matrix.Rotation(math.radians(rot_degree[0]), 4, 'X') @ mathutils.Matrix.Rotation(math.radians(rot_degree[1]), 4, 'Y') @ mathutils.Matrix.Rotation(math.radians(rot_degree[2]), 4, 'Z')

    obj.matrix_world = obj.matrix_world @ rotation_matrix
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = 'ACTIVE'

    obj.rigid_body.mass = 1.0
    
    # Increase damping for faster resting
    obj.rigid_body.linear_damping = 0.8  # Higher values cause faster linear slowdown
    obj.rigid_body.angular_damping = 0.9  # Higher values cause faster angular slowdown
    
    # Increase friction to reduce sliding
    obj.rigid_body.friction = 1.0  # Max value is 1.0
    
    # Lower bounciness (restitution)
    obj.rigid_body.restitution = 0.0 

    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.select_all(action='DESELECT')
    plane.select_set(True)
    bpy.ops.rigidbody.object_add()
    plane.rigid_body.type = 'PASSIVE'

    bpy.ops.ptcache.bake_all(bake=True)

    #scene = bpy.context.scene
    #scene.frame_set(num_frames)

    scene = bpy.data.scenes['Scene']
    frame_info = []

    import copy
    for frame in range(scene.frame_start,
                       scene.frame_end,
                       scene.frame_step):

        scene.frame_set(frame)
        frame_info.append(copy.deepcopy(obj.matrix_world))

    stable_pose = np.array(frame_info[-1])[:3,:3]

    # bpy.ops.wm.save_as_mainfile(filepath='/home/s2514643/MOAD-data-generation-latest/MOAD-v1/ires.blend')

    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.ops.object.select_by_type(type='OBJECT')
    # bpy.ops.object.delete()

    return stable_pose


def minimum_bounding_area(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    
    # Calculate the lengths of the sides of the bounding box
    side_lengths = max_coords - min_coords
    
    # Calculate the area of the bounding box
    area = 2 * (side_lengths[0] * side_lengths[1] +
                side_lengths[0] * side_lengths[2] +
                side_lengths[1] * side_lengths[2])
    
    return area
    
def re_pose_blend_file(blend_file_path, out_blend_file_path, plane_height = 2, rot_degree = 0, num_frames = 250):

    rots = simulate_to_get_stable_pose(blend_file_path, plane_height = plane_height, rot_degree = rot_degree, num_frames = num_frames)
    load_blend_file(blend_file_path)
    join_all_objects()
    obj = bpy.data.objects[0]

    matrix = np.eye(4,4)
    matrix[:3,:3] = rots.T
    obj.matrix_world = matrix
    bpy.ops.wm.save_as_mainfile(filepath=out_blend_file_path)

    with open(out_blend_file_path.replace('.blend', '.json'), 'w') as json_file:
        json.dump({"simulated_stable_pose": matrix.tolist()}, json_file, indent=2)

    if os.path.isfile(blend_file_path.replace('.blend', '.ply')):
        pont_cloud_label = trimesh.load(blend_file_path.replace('.blend', '.ply'))
        pont_cloud_label.apply_transform(matrix.T)
        pont_cloud_label.export(out_blend_file_path.replace('.blend', '.ply'))

    if os.path.isfile(blend_file_path.replace('.blend', '.obj')):
        mesh_label = trimesh.load(blend_file_path.replace('.blend', '.obj'))
        mesh_label.apply_transform(matrix.T)
        mesh_label.export(out_blend_file_path.replace('.blend', '.obj'))

def get_scene_folder_path(output_folder, model_name):
    #os.makedirs(output_folder, exist_ok = True)
    index = 1
    while True:
        folder_path = os.path.join(output_folder, f'{model_name}-{index:03d}')
        if not os.path.exists(folder_path):
            return folder_path
        index += 1

def get_file_path(output_folder, model_name, mode):
    os.makedirs(output_folder, exist_ok = True)
    index = 1
    while True:
        file_path = os.path.join(output_folder, f'{model_name}-{mode}-{index:03d}.blend')
        if not os.path.exists(file_path):
            return file_path
        index += 1

def get_file_path_obj(output_folder, model_name, mode):
    os.makedirs(output_folder, exist_ok = True)
    index = 1
    while True:
        file_path = os.path.join(output_folder, f'{model_name}-{mode}-{index:03d}.obj')
        if not os.path.exists(file_path):
            return file_path
        index += 1


def optics_clustering(blend_file):
    load_blend_file(blend_file)
    original_mesh = load_scene_as_obj()
    clustering = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.01)
    n_connected_comps = len(np.unique(clustering.fit( trimesh.sample.sample_surface_even(original_mesh, 10000)[0]).labels_))
    return n_connected_comps

def get_connected_components(mesh, eps = 0.5):
    dbc = skc.DBSCAN(eps = eps)
    n_connected_comps = len(np.unique(dbc.fit( trimesh.sample.sample_surface(mesh, 5000)[0]).labels_))
    return n_connected_comps

def load_scene_as_obj():
    temp_path = tempfile.NamedTemporaryFile().name+'.obj'
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.export_scene.obj(
        filepath=temp_path,
        use_selection=True,
        use_materials=False,  # Set to True if you want to include material information
        use_triangles=True ,
        axis_forward='Y',
        axis_up='Z'   # Set to False if you want to export quads instead of triangles
    )
    mesh = trimesh.load(temp_path, force = 'mesh')
    os.remove(temp_path)
    return mesh

def load_object_as_obj(obj):

    output_path = tempfile.NamedTemporaryFile().name+'.obj'
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.obj(filepath=output_path, use_selection=True, use_materials=False, use_triangles=True , axis_forward='Y',
                             axis_up='Z')
    mesh = trimesh.load(output_path, force = 'mesh')
    os.remove(output_path)
    return mesh

def get_mesh_info_from_obj(obj): ## obj -> v,f,mat
    v = []  
    f = []  
    if obj.type == 'MESH':
        mesh = obj.data
        for vert in mesh.vertices:
            v.append(vert.co[:])
        for face in mesh.polygons:
            f.append(face.vertices[:])

    obj_name = obj.name
    mats = obj.material_slots
    return v, f, mats, obj_name

def get_watertight_mesh(query_mesh, form_cage = True):
    v, f = query_mesh.vertices, query_mesh.faces
    mfix = PyTMesh(False)
    mfix.load_array(v, f)
    mfix.fill_small_boundaries(nbe=100, refine=True)
    cv, cf = mfix.return_arrays()
    if form_cage : cv, cf = lazy_cage(cv, cf, grid_size = 256, num_faces = 1000)
    query_mesh = trimesh.Trimesh(vertices = cv, faces = cf)
    return query_mesh
    
def make_triangular_mesh():
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')

def extract_watertight_regions(min_region, max_region):

    filted_meshes = []
    all_meshes= []
    all_mats = []
    for query_obj in bpy.context.scene.objects:
        v, f, mats, obj_name = get_mesh_info_from_obj(query_obj)
        try:
            mesh = trimesh.Trimesh(vertices=np.array(v), faces=np.array(f))
            all_meshes.append([mesh, mats, obj_name])
            all_mats.append(mats)
        except:
            print ('WRN: error creating trimesh mesh. ignoring...')
            return [], [], []

    total_area = sum(trimesh.util.concatenate([i[0] for i in all_meshes]).bounding_box.area_faces)

    for mesh, mats, obj_name in all_meshes:

        mesh_portion_area = sum(mesh.bounding_box.area_faces)
        n_connected_comps = get_connected_components(mesh)
        # print (n_connected_comps, mesh_portion_area)
        if (mesh_portion_area/total_area)>min_region and \
            (mesh_portion_area/total_area)<=max_region and \
                n_connected_comps == 1:
            if mesh.is_watertight:
                filted_meshes.append([mesh, mats, obj_name, True])
            else:
                mfix = PyTMesh(False)
                mfix.load_array(mesh.vertices, mesh.faces)
                mfix.fill_small_boundaries(nbe=100, refine=True)
                vert, faces = mfix.return_arrays()
                meshfix = trimesh.Trimesh(vertices = vert, faces = faces)
                if meshfix.is_watertight:
                    print ('log: converted to watertight mesh.')
                    filted_meshes.append([meshfix, mats, obj_name, False])

    print (f'log: total {len(filted_meshes)} segments found.')
    return filted_meshes, all_mats, all_meshes
    
    # out_mesh_data_list = []
    # for mesh_data, mats, obj_name, is_watertight in filted_meshes:
    #     # wt_meshes = []
    #     # mesh_data = [i for i in mesh_data if len(trimesh.graph.connected_components(i.face_adjacency, min_len=3))==1]
    #     if len(mesh_data) == 0: 
    #         continue
    #     mesh_i = mesh_data[np.argmax([sum(i.bounding_box.area_faces) for i in mesh_data])]
    #     mesh_portion_area = sum(mesh_i.bounding_box.area_faces)
    #     if mesh_i.is_watertight and \
    #         (mesh_portion_area/total_area)>min_region and \
    #         (mesh_portion_area/total_area)<=max_region:
    #         # wt_meshes.append(mesh_i)
    #         chosen_mesh = mesh_i
    #         mesh_data_sepr = {'candidate':chosen_mesh, 'others':list(set(mesh_data)-{chosen_mesh})}
    #         out_mesh_data_list.append([mesh_data_sepr,mats,obj_name, is_watertight])
    #     # for mesh_i in mesh_data:
    #     #     mesh_portion_area = sum(mesh_i.bounding_box.area_faces)
    #     #     print (mesh_portion_area)
    #     #     if mesh_i.is_watertight and \
    #     #         (mesh_portion_area/total_area)>min_region and \
    #     #         (mesh_portion_area/total_area)<=max_region:
    #     #         wt_meshes.append(mesh_i)
    #     # if len(wt_meshes)>0:
    #     #     chosen_mesh = random.choice(wt_meshes)
    #     #     mesh_data_sepr = {'candidate':chosen_mesh, 'others':list(set(mesh_data)-{chosen_mesh})}
    #     #     out_mesh_data_list.append([mesh_data_sepr,mats,obj_name, is_watertight])
    
    # # filted_meshes = [i for i in filted_meshes if (i[0].vertices.shape[0]/n_verts)>min_region and \
    # #                                 (i[0].vertices.shape[0]/n_verts)<=max_region]

    
    # return out_mesh_data_list

def create_obj_from_mesh(v,f,mats = None, name = 'new_object'): ## v,f,mat -> obj

    new_mesh = bpy.data.meshes.new('new_mesh')
    new_mesh.from_pydata(v, [], f)
    new_mesh.update()
    new_object = bpy.data.objects.new(name, new_mesh)
    if mats is not None:
        for material_slot in mats:
            material = material_slot.material
            new_material = material.copy()
            new_object.data.materials.append(new_material)

    return new_object

def load_blend_file(file_path):
    bpy.ops.wm.open_mainfile(filepath=file_path)
    bpy.context.view_layer.objects.active = bpy.data.objects[0]

def seperate_objects_based_on_materials():
    active_object = bpy.context.view_layer.objects.active
    if active_object and active_object.type == 'MESH' and bpy.context.mode == 'OBJECT':
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='MATERIAL')
        bpy.ops.object.mode_set(mode='OBJECT')

def seperate_objects_based_on_loose_parts():
    active_object = bpy.context.view_layer.objects.active
    if active_object and active_object.type == 'MESH' and bpy.context.mode == 'OBJECT':
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')

def join_all_objects():

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[-1]
    bpy.ops.object.join()
    bpy.context.active_object.name = "object"

def apply_anomaly(mesh):
    return mesh

def rotation_matrix_4x4(axis, theta):
    """
    Calculate 4x4 rotation matrix for small angle rotation using Rodrigues' formula.
    
    Parameters:
    - axis: Unit vector representing the rotation axis [x, y, z]
    - theta: Small angle of rotation in radians
    
    Returns:
    - 4x4 rotation matrix
    """
    kx, ky, kz = axis
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    rotation_3x3 = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    rotation_4x4 = np.eye(4)
    rotation_4x4[:3, :3] = rotation_3x3
    
    return rotation_4x4


def rotation_matrix_4x4_center(center, axis, angle):
    # Convert axis to unit vector
    axis = np.array(axis) / np.linalg.norm(axis)

    # Create a rotation matrix using scipy's Rotation class
    rotation_matrix = Rotation.from_rotvec(angle * axis).as_matrix()

    # Create a 4x4 transformation matrix with rotation
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix

    # Set the center of rotation
    transformation_matrix[:3, 3] = center - np.dot(rotation_matrix, center)

    return transformation_matrix
# # Example usage for a small angle rotation around the z-axis
# axis_of_rotation = np.array([0, 0, 1])  # Rotation around the z-axis
# rotation_angle = np.radians(10)  # Small angle in radians


def merge_objects(obj1, obj2, new_obj_name="MergedObject"):
    # Duplicate the objects to preserve the originals
    bpy.ops.object.select_all(action='DESELECT')
    # obj1.select_set(True)
    bpy.context.view_layer.objects.active = obj1
    bpy.ops.object.duplicate(linked=False)
    duplicate_obj1 = bpy.context.active_object

    bpy.ops.object.select_all(action='DESELECT')
    # obj2.select_set(True)
    bpy.context.view_layer.objects.active = obj2
    bpy.ops.object.duplicate(linked=False)
    duplicate_obj2 = bpy.context.active_object

    # Join the duplicate objects into a new one
    bpy.ops.object.select_all(action='DESELECT')
    duplicate_obj1.select_set(True)
    duplicate_obj2.select_set(True)
    bpy.context.view_layer.objects.active = duplicate_obj1
    bpy.ops.object.join()

    # Rename the new object
    bpy.context.active_object.name = new_obj_name

    return bpy.context.active_object

def random_point_on_surface(mesh):
    # Select a random face
    random_face_index = np.random.choice(len(mesh.faces))

    # Get the vertices of the selected face
    face_vertices = mesh.vertices[mesh.faces[random_face_index]]

    # Calculate barycentric coordinates
    u, v, _ = np.random.uniform(0, 1, size=3)
    while u + v > 1:
        u, v, _ = np.random.uniform(0, 1, size=3)

    # Calculate the random point on the face
    random_point = (1 - u - v) * face_vertices[0] + u * face_vertices[1] + v * face_vertices[2]

    return random_point



def get_normal_at_point(mesh, point):
    # Find the closest point on the mesh to the given point
    _, index = mesh.kdtree.query(point)

    # Get the normal vector at the closest point
    normal = mesh.face_normals[mesh.vertex_faces[index][0]]

    return normal

def is_intersect(m1, m2):
    return gpytoolbox.copyleft.do_meshes_intersect(m1.vertices,m1.faces,m2.vertices,m2.faces)


def random_subset_box(main_box):
    
    min_point = np.random.uniform(main_box[0], main_box[1])
    max_point = np.random.uniform(min_point, main_box[1])

    subset_box = (tuple(min_point), tuple(max_point))
    return np.array(subset_box)


def is_box_enclosing_mesh(obj_vertices, bounding_box_min, bounding_box_max):
    mesh_min = np.min(obj_vertices, axis=0)
    mesh_max = np.max(obj_vertices, axis=0)

    for axis in range(3):  # Check for each axis (x, y, z)
        if mesh_min[axis] < bounding_box_min[axis] or mesh_max[axis] > bounding_box_max[axis]:
            return False
    return True



def check_connected_component_fine(mesh):
    voxel = creation.voxelize(mesh, pitch = 0.05)
    num_ = len(np.unique(cc3d.connected_components(np.array(voxel.matrix, dtype = np.uint8))))

    if num_ >2:
        return False
    else:
        return True

def calculate_triangle_area(vertex1, vertex2, vertex3):
    # Calculate the area of a triangle given its three vertices
    edge1 = np.array(vertex2) - np.array(vertex1)
    edge2 = np.array(vertex3) - np.array(vertex1)

    cross_product = np.cross(edge1, edge2)
    area = 0.5 * np.linalg.norm(cross_product)

    return area

def calculate_mesh_surface_area(vertices, faces):
    # Calculate the total surface area of the mesh
    total_area = 0.0

    for face in faces:
        vertex1 = vertices[face[0]]
        vertex2 = vertices[face[1]]
        vertex3 = vertices[face[2]]

        total_area += calculate_triangle_area(vertex1, vertex2, vertex3)

    return total_area

def get_random_transformation_matrix(exclude_mesh, include_mesh, mode, min_trans, max_trans, min_angle, max_angle, min_scale, max_scale, max_try):

    for _ in range(max_try):

        mesh1 = exclude_mesh.copy() # will apply transformation on this mesh
        mesh2 = include_mesh.copy() 

        if mode == 'tran':
            trans_xyz = np.array([np.random.uniform(min_trans, max_trans)*random.choice([-1,1]) for i in range(3)])
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, 3] = trans_xyz

        if mode == 'rota':
            manager = trimesh.collision.CollisionManager()
            manager.add_object('mesh', mesh2)  
            contact_points_list = manager.in_collision_single(mesh1, return_names=True, return_data=True)[-1]
            contact_point = np.random.choice(contact_points_list).point
            center_of_rotation = contact_point
            rotation_axis = random.choice([np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])])
            rotation_angle = np.random.uniform(min_angle, max_angle)*random.choice([-1,1])
            transformation_matrix = rotation_matrix_4x4_center(center_of_rotation, rotation_axis, rotation_angle)

        if mode == 'scal':
            scale_factors = np.array([np.random.choice([np.random.uniform(min_scale, max_scale), np.random.uniform(2-max_scale, 2-min_scale)]) for _ in range(3)])
            transformation_matrix = np.eye(4)
            np.fill_diagonal(transformation_matrix, scale_factors)

        output_mesh = trimesh.util.concatenate([mesh1.apply_transform(transformation_matrix), mesh2])

        if check_connected_component_fine(output_mesh) and get_connected_components(output_mesh) == 1:
            return transformation_matrix 

    return False


def random_increase_dimensions(bounding_box, scale_factor_range=(1.1, 2.0)):
    # Generate random scale factors for each axis
    scale_factors = np.random.uniform(scale_factor_range[0], scale_factor_range[1], size=3)

    # Calculate the center of the bounding box
    center = 0.5 * (bounding_box[0] + bounding_box[1])

    # Calculate the dimensions of the bounding box
    dimensions = bounding_box[1] - bounding_box[0]

    # Increase the dimensions based on the random scale factors
    increased_dimensions = dimensions * scale_factors

    # Calculate the new bounding box
    new_bounding_box = np.vstack((center - 0.5 * increased_dimensions, center + 0.5 * increased_dimensions))

    return new_bounding_box

def get_random_mesh_parts(splits, obj_names, min_region, max_region, num_iters, total_area, connection_check = 'include'):
    if len(splits) == 0:
        return []
    for _ in range(num_iters):
        initial_dimesion = random.choice(splits).bounding_box.bounds
        subset_bbox = random_increase_dimensions(initial_dimesion, scale_factor_range=(1.1, 2.0))
        
        excludes = []
        includes = []
        excludes_names = []
        includes_names = []
        for split, name in zip(splits, obj_names):

            if is_box_enclosing_mesh(split.vertices, subset_bbox[0], subset_bbox[1]) and split.vertices.shape[0]>5:
                excludes.append(split)
                excludes_names.append(name)
            else:
                includes.append(split)
                includes_names.append(name)
            
        if len(excludes)>0 and len(includes)>0:
            exclude_mesh = trimesh.util.concatenate(excludes)
            exclude_area = calculate_mesh_surface_area(exclude_mesh.vertices, exclude_mesh.faces) #sum(exclude_mesh.bounding_box.area_faces)
            if (exclude_area/total_area)>min_region and \
                (exclude_area/total_area)<=max_region:
                include_mesh = trimesh.util.concatenate(includes)
                if connection_check == 'include':
                    if get_connected_components(include_mesh, eps = 0.5) == 1:
                        if check_connected_component_fine(include_mesh):
                            return [excludes, includes, excludes_names, includes_names, exclude_area]
                elif connection_check == 'exclude':
                    if get_connected_components(exclude_mesh, eps = 0.5) == 1:
                        if check_connected_component_fine(exclude_mesh):
                            return [excludes, includes, excludes_names, includes_names, exclude_area]
    return []

def get_mesh_data():

    all_mesh_data = {}
    all_mesh_splits = []
    all_obj_names = []

    for query_obj in bpy.context.scene.objects:
        v, f, mats, obj_name = get_mesh_info_from_obj(query_obj)
        try:
            mesh = trimesh.Trimesh(vertices=np.array(v), faces=np.array(f))
            all_mesh_data[obj_name] = mats #.append([mats, obj_name])
            mesh_splits = list(mesh.split(only_watertight=True))
            if len(mesh_splits) == 0:
                mesh_splits = [mesh]
            all_mesh_splits.extend(mesh_splits)
            all_obj_names.extend([obj_name]*len(mesh_splits))
        except:
            print ('WRN: error creating trimesh mesh. ignoring...')

    return all_mesh_splits, all_obj_names, all_mesh_data


def join_selected_objects(object_name_list, joined_object_name):

    if len(object_name_list) == 0:
        return None

    bpy.ops.object.select_all(action='DESELECT')
    for obj_name in object_name_list:
        bpy.data.objects[obj_name].select_set(True)

    # Join the selected objects
    bpy.context.view_layer.objects.active = bpy.data.objects[object_name_list[0]]
    bpy.ops.object.join()

    # Rename the joined object
    bpy.context.active_object.name = joined_object_name

def cd_distance(mesh1, mesh2):
    distances_mesh1_to_mesh2 = mesh1.nearest.on_surface(mesh2.vertices)[1]
    distances_mesh2_to_mesh1 = mesh2.nearest.on_surface(mesh1.vertices)[1]
    
    chamfer_distance = np.mean(distances_mesh1_to_mesh2) + np.mean(distances_mesh2_to_mesh1)
    return chamfer_distance

def bounding_box_area(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    
    # Calculate the lengths of the sides of the bounding box
    side_lengths = max_coords - min_coords
    
    # Calculate the area of the bounding box
    area = side_lengths[0] * side_lengths[1] * side_lengths[2]
    
    return area

def create_point_label(input_mesh, deform_mesh, eps = 0.1):
    pnts2 = trimesh.sample.sample_surface(input_mesh, 10000)[0]
    pnts1 =trimesh.sample.sample_surface(deform_mesh, 10000)[0]
    distances = scipy.spatial.distance.cdist(pnts1, pnts2)
    min_distances_pt1 = np.min(distances, axis=1)
    diff_pt1_indices = np.where(min_distances_pt1 > eps)[0]
    point_cloud = trimesh.PointCloud(pnts1, colors = [255,255,255,255])
    point_cloud.colors[diff_pt1_indices] = [0,0,0,0]
    return point_cloud

def create_point_label_segments(include_mesh, transformed_exclude_mesh):


    # include_mesh.visual.vertex_colors = np.array([[255,255,255]]*len(include_mesh.vertices))
    # transformed_exclude_mesh.visual.vertex_colors = np.array([[255,0,0]]*len(transformed_exclude_mesh.vertices))
    output_mesh = trimesh.util.concatenate([include_mesh,transformed_exclude_mesh])

    pnts1 = trimesh.sample.sample_surface_even(include_mesh, 20000, radius = 0.04)[0]
    point_cloud1 = trimesh.PointCloud(pnts1, colors = [255,255,255])

    if transformed_exclude_mesh != []:

        pnts2 = trimesh.sample.sample_surface_even(transformed_exclude_mesh, 20000, radius = 0.04)[0]
        point_cloud2 = trimesh.PointCloud(pnts2, colors = [0,0,0])

        concatenated_vertices = np.vstack([point_cloud1.vertices, point_cloud2.vertices])
        concatenated_colors = np.vstack([point_cloud1.colors, point_cloud2.colors])
        concatenated_point_cloud = trimesh.PointCloud(vertices=concatenated_vertices, colors=concatenated_colors)
        
        return concatenated_point_cloud, output_mesh

    return point_cloud1, output_mesh


def convert_and_save_as_joined(path):
    load_blend_file(path)
    join_all_objects()
    bpy.ops.wm.save_as_mainfile(filepath=path)

def modify_vertex_coordinates(mesh, target_coords, new_coords, tolerance=0.0000001):
    for vertex in mesh.vertices:
        if math.isclose(vertex.co.x, target_coords[0], rel_tol=tolerance) and \
           math.isclose(vertex.co.y, target_coords[1], rel_tol=tolerance) and \
           math.isclose(vertex.co.z, target_coords[2], rel_tol=tolerance):
            # Modify the coordinates of the found vertex
            vertex.co = new_coords
            return True
    return False

def seperate_portions_(obj_data, exclude_mesh):
    #alterd_index = (np.where(~(whole_mesh.vertices == out_mesh.vertices).all(1)))[0]

    bpy_verts_idxs = []

    obj_data_verts = np.array([i.co for i in obj_data.vertices])
    obj_query_verts = np.array(exclude_mesh.vertices)
    matches = np.all(obj_query_verts[:, None, :] == obj_data_verts[None, :, :], axis=2)
    bpy_verts_idxs = np.argwhere(matches)[:,1]

    #for exc_verts in exclude_mesh.vertices:
    #    bpy_verts_idxs.extend([i.index for i in obj_data.vertices if (exc_verts==i.co).all()])
    
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    for v_index in bpy_verts_idxs:
        obj_data.vertices[v_index].select = True
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    return True


def seperate_portions_approx_(obj_data, exclude_mesh):
    #alterd_index = (np.where(~(whole_mesh.vertices == out_mesh.vertices).all(1)))[0]

    bpy_verts_idxs = []
    for exc_verts in exclude_mesh.vertices:
        print (len(bpy_verts_idxs))
        bpy_verts_idxs.extend([i.index for i in obj_data.vertices if sum((exc_verts-i.co)**2)<0.1])
    

    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    for v_index in bpy_verts_idxs:
        obj_data.vertices[v_index].select = True
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode = 'OBJECT')

    return True


def boolean_difference(obj_to_cut, mesh, ops = 'DIFFERENCE'):

    start_time = time.time()

    scene = bpy.context.scene

    obj_copy = obj_to_cut.copy()
    obj_copy.data = obj_to_cut.data.copy()
    scene.collection.objects.link(obj_copy)
    obj_copy.name = ops

    obj_cutter = create_obj_from_mesh(mesh.vertices.tolist(), mesh.faces.tolist())
    scene.collection.objects.link(obj_cutter)

    bpy.context.view_layer.objects.active = obj_cutter
    disp_mod = obj_cutter.modifiers.new("Displace", type="DISPLACE")
    disp_mod.mid_level = 0.85  # Midpoint around which displacement occurs
    bpy.ops.object.modifier_apply({"object": obj_cutter}, modifier="Displace")

    bpy.context.view_layer.objects.active = obj_copy

    obj_cutter.select_set(True)
    obj_copy.select_set(True)

    # Apply boolean difference
    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers['Boolean'].operation = ops
    bpy.context.object.modifiers['Boolean'].use_self = True
    bpy.context.object.modifiers['Boolean'].object = obj_cutter

    bpy.ops.object.modifier_apply({"object": obj_copy}, modifier="Boolean")

    bpy.data.objects.remove(obj_cutter, do_unlink=True)
    bpy.ops.object.select_all(action='DESELECT')

    print (f'Time taken boolean operation : {int(time.time()-start_time)}')



# def boolean_intersect(obj_to_cut, mesh, ops = 'INTERSECT'):

#     start_time = time.time()

#     scene = bpy.context.scene

#     obj_copy = obj_to_cut.copy()
#     obj_copy.data = obj_to_cut.data.copy()
#     scene.collection.objects.link(obj_copy)

#     obj_cutter = create_obj_from_mesh(mesh.vertices.tolist(), mesh.faces.tolist())
#     scene.collection.objects.link(obj_cutter)
#     obj_cutter.name = ops

#     bpy.context.view_layer.objects.active = obj_cutter
#     disp_mod = obj_cutter.modifiers.new("Displace", type="DISPLACE")
#     disp_mod.mid_level = 0.85  # Midpoint around which displacement occurs
#     bpy.ops.object.modifier_apply({"object": obj_cutter}, modifier="Displace")

#     bpy.context.view_layer.objects.active = obj_cutter

#     obj_cutter.select_set(True)
#     obj_copy.select_set(True)

#     # Apply boolean difference
#     bpy.ops.object.modifier_add(type='BOOLEAN')
#     bpy.context.object.modifiers['Boolean'].operation = ops
#     bpy.context.object.modifiers['Boolean'].use_self = True
#     bpy.context.object.modifiers['Boolean'].object = obj_copy

#     bpy.ops.object.modifier_apply({"object": obj_cutter}, modifier="Boolean")

#     bpy.data.objects.remove(obj_copy, do_unlink=True)
#     bpy.ops.object.select_all(action='DESELECT')

#     print (f'Time taken boolean operation : {int(time.time()-start_time)}')


# def get_stable_pose_from_blend_file(blend_file_path):
#     load_blend_file(blend_file_path)
#     mesh = load_scene_as_obj()
#     stable_pose_estimate = mesh.compute_stable_poses()
#     return stable_pose_estimate[0][0]