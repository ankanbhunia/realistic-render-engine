import bpy
import numpy as np
import colorsys
import json
import os
import shutil

from mathutils import Matrix, Vector

# TOYS_ROOT = '/disk/nfs/gazinasvolume1/datasets/toys4k_blend_files/'

def get_id_info(path, dataset_type):
    if dataset_type == "modelnet":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-1][:-4]
        return category, obj_id

    if dataset_type == "shapenet":
        category = path.split("/")[-4]
        obj_id = path.split("/")[-3]
        return category, obj_id

    if dataset_type == "toys":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-1]
        return category, obj_id.split('.blend')[0]
    
    if dataset_type == "ABC":
        obj_id = path.split("/")[-1]
        category = ''

        return category, obj_id
    
    if dataset_type == "toys200":
        obj_id = path.split('/')[-1]
        category = ""
        return category, obj_id

    if dataset_type == "clever":
        obj_id = path.split('/')[-1]
        category = ""
        return category, obj_id

def load_obj(scn, path, dataset_type):
    
    category, obj_name = get_id_info(path, dataset_type)

    if dataset_type == "toys200":
        path = '/'.join(path.split('/')[:-1])
        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects if name == obj_name]
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        for obj in data_to.objects:
            if obj is not None:
                scn.collection.objects.link(obj)


        bpy.context.active_object.name = obj_name
        bpy.data.objects[obj_name].name = obj_name

        obj = bpy.data.objects[obj_name]

        obj.hide_viewport = False
        obj.hide_render = False
        obj.rotation_mode = "XYZ"

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        obj.rotation_euler = (np.radians(90), 0, 0)

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "toys":

        obj = add_toys4k_obj(scn, path, category, obj_name, 1, (0,0))


        bpy.context.view_layer.objects.active = obj
        obj.rotation_mode = "XYZ"

        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "modelnet":
        bpy.ops.import_scene.obj(
            filepath=path,
            axis_forward="Y",
            axis_up="Z")

        obj_names = [obj.name for obj in bpy.data.objects]
        blender_obj_name = [x for x in sorted(obj_names) if obj_name in x][-1]
        obj = bpy.data.objects[blender_obj_name]

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "shapenet":
        bpy.ops.import_scene.obj(
            filepath=path,
            axis_forward="Y",
            axis_up="Z")

        objs = [obj for obj in bpy.data.objects if obj.name.startswith('model_normalized')]

        for obj in objs:
            if obj is not None:
                scn.collection.objects.link(obj)
                current_name = obj.name
                bpy.data.objects[current_name].name = obj_name

        obj = bpy.data.objects[obj_name]
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (np.radians(90), 0, 0)

        # Set the new object as active, then rotate, scale, and translate it
        x, y = (0,0)
        scale = 1
        theta = 0
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.context.object.rotation_euler[2] = theta
        bpy.ops.transform.resize(value=(scale, scale, scale))
        bpy.ops.transform.translate(value=(x, y, scale))

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "clever":
        object_dir = "../clever/obj"
        obj = add_clvr_object(object_dir, obj_name, 1, (0,0))
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        #### Randomize material
        material_dir = "../clever/material"
        load_clvr_materials(material_dir)
        rint = np.random.randint(2)
        if rint == 0:
            mat = "Rubber"
        else:
            mat = "MyMetal"
        rgba = np.random.rand(4)
        rgba[-1] = 1.0
        add_clvr_material(mat, Color=rgba)

        return obj

    if dataset_type == "ABC":
        obj_name = path.split('/')[-1].replace('.glb', '.obj')
        
        # path = path.replace('.obj', '.glb')
        # path = path.replace('/objects', '')
        print (path)
        bpy.ops.import_scene.obj(filepath=path)
        print (list(bpy.data.objects))
        bpy.ops.object.select_all(action='DESELECT')

        #bpy.data.objects['world'].select_set(True) # Blender 2.8x

        bpy.ops.object.delete() 
        print ([obj.name for obj in bpy.data.objects],obj_name[:-4])
        obj = [obj for obj in bpy.data.objects if (obj.name == obj_name[:-4])][0]
        obj.rotation_mode = "XYZ"
        
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)


        return obj

def fix_materials(materials):
    # sometimes we don't cast shadows, and this should fix that
    for mat in bpy.data.materials:

        if not mat.use_nodes:
            continue 

        tree = mat.node_tree
        nodes = mat.node_tree.nodes

        has_img_node = len([x for x in nodes if x.label == 'BASE COLOR']) == 1
        has_light_path_node  = len([x for x in nodes if x.name == 'Light Path']) == 1

        if has_img_node and has_light_path_node:
            img_node = nodes['Image Texture']
            output_node = nodes['Material Output']

            nodes.new(type = 'ShaderNodeBsdfPrincipled')
                
            BSDF = nodes['Principled BSDF']

            output = BSDF.inputs['Base Color']
            inpt = img_node.outputs['Color']
            tree.links.new(inpt, output)

            inpt = nodes["Material Output"].inputs['Surface']
            output = BSDF.outputs[0]

            tree.links.new(inpt, output)


def add_plane(coords, size):

    bpy.ops.mesh.primitive_plane_add(
            size=size, 
            enter_editmode=False, 
            align='WORLD', 
            location=coords, 
            scale=(1, 1, 1))

    obj = bpy.data.objects['Plane']
    obj.name = 'floor_object'

def add_texture_node(nodes, name):
    
    nodes.new(type="ShaderNodeTexImage")

    node = nodes['Image Texture']
    node.name = name
    node.label = name

    return node

def add_cube(coords, rotation, scale):
    bpy.ops.mesh.primitive_cube_add(
            enter_editmode=False, 
            align='WORLD', 
            location=coords, 
            rotation=rotation,
            scale=(scale, scale, scale))

def add_sphere(coords, rotation, scale):
    bpy.ops.mesh.primitive_uv_sphere_add(
            segments=64, 
            ring_count=64, 
            location=coords, 
            rotation=rotation,
            radius=scale/2)

def add_cylinder(coords, rotation, scale):
    bpy.ops.mesh.primitive_uv_sphere_add(
            segments=64, 
            ring_count=64, 
            location=coords, 
            rotation=rotation,
            radius=scale)

def add_cubes(n):
    
    obj_list = []
    for i in range(n):
        
        scale = np.random.uniform(0.05, 0.15)

        add_cube((0,0,0), (0, 0, 0), scale)
        obj = bpy.data.objects['Cube']
        obj.name = 'Cube_{}'.format(i)
        bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
        
        mat_name = 'cube_mat_{}'.format(i)
        mat = add_principled_material(bpy.data, mat_name)
        obj.data.materials.append(bpy.data.materials[mat_name])

        obj_list.append(obj.name)

    return obj_list

def add_spheres(n):
    
    obj_list = []
    for i in range(n):
        
        scale = np.random.uniform(0.05, 0.15)

        add_sphere((0,0,0), (0, 0, 0), scale)
        obj = bpy.data.objects['Sphere']
        obj.name = 'Sphere_{}'.format(i)
        bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
        
        mat_name = 'sphere_mat_{}'.format(i)
        mat = add_principled_material(bpy.data, mat_name)
        obj.data.materials.append(bpy.data.materials[mat_name])

        obj_list.append(obj.name)

    return obj_list

def add_configured_material(data, mat_dct):
    mat_type = mat_dct['type']
    mat_clever = mat_dct['clever']

    data.materials.new(mat_type)
    mat = bpy.data.materials[mat_type]
    mat.use_nodes = True

    tree = mat.node_tree
    nodes = tree.nodes
        
    # main BSDF node
    nodes.new(type = 'ShaderNodeBsdfPrincipled')
    BSDF = nodes['Principled BSDF']
    BSDF.inputs['Base Color'].default_value = (0, 0, 0, 1)

    BSDF.inputs['Specular'].default_value = mat_dct['specular']
    BSDF.inputs['Roughness'].default_value = mat_dct['roughness'] 
    
    input = nodes["Material Output"].inputs['Surface']
    output = BSDF.outputs[0]

    tree.links.new(input, output)

    if mat_type in ['random-mono', 'random-multi']:
        color_ramp = nodes.new(type="ShaderNodeValToRGB")
        colors = mat_dct['color']

        for i in np.arange(len(colors)):
            color_ramp.color_ramp.elements.new((1/(1+len(colors)))*(i+1))
                    
        for i, (r,g,b) in enumerate(colors):
            color_ramp.color_ramp.elements[i].color = (r,g,b,1)

        color_ramp.color_ramp.interpolation = 'CONSTANT'

        nodes.new(type = 'ShaderNodeTexVoronoi')
        tex = nodes['Voronoi Texture']
        tex.inputs['Randomness'].default_value = mat_dct['randomness']
        tex.inputs['Scale'].default_value = mat_dct['scale']

        output = tex.outputs['Color']
        input = color_ramp.inputs['Fac']

        tree.links.new(output, input)

        output = color_ramp.outputs['Color']
        input = BSDF.inputs['Base Color']

        tree.links.new(output, input)

    if mat_type == "single":
        BSDF.inputs['Base Color'].default_value = (
            mat_dct['color'][0],
            mat_dct['color'][1],
            mat_dct['color'][2],
            1)

    if mat_clever:
        load_clvr_materials("../clever/material")
        group_node = nodes.new('ShaderNodeGroup')
        group_node.node_tree = bpy.data.node_groups["MyMetal"]

        tree.links.new(
            group_node.outputs['Shader'],
            nodes["Material Output"].inputs['Surface']
        )

        if mat_type in ['random-mono', 'random-multi']:

            tree.links.new(
                color_ramp.outputs['Color'],
                group_node.inputs['Color']
            )
        elif mat_type == "single":
            group_node.inputs['Color'].default_value = (
            mat_dct['color'][0],
            mat_dct['color'][1],
            mat_dct['color'][2],
            1)
    
    return mat_type

def add_principled_material(data, name):
        
    assert isinstance(name, str)

    data.materials.new(name)
    mat = bpy.data.materials[name]
    mat.use_nodes = True
   
    h = np.random.uniform(0,1)
    s = np.random.uniform(0.4,1)
    l = np.random.uniform(0.4,0.6)
    
    r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]

    tree = mat.node_tree
    nodes = tree.nodes
    nodes.new(type = 'ShaderNodeBsdfPrincipled')
    
    BSDF = nodes['Principled BSDF']
    BSDF.inputs['Base Color'].default_value = (r, g, b, 1)

    BSDF.inputs['Specular'].default_value = np.random.uniform(0.6,0.9)
    BSDF.inputs['Roughness'].default_value = np.random.uniform(0.1,0.25)
    inpt = nodes["Material Output"].inputs['Surface']
    output = BSDF.outputs[0]

    tree.links.new(inpt, output)

    return mat

def add_principled_material_test(data, name):
        
    assert isinstance(name, str)

    data.materials.new(name)
    mat = bpy.data.materials[name]
    mat.use_nodes = True
   
    h = np.random.uniform(0,1)
    s = np.random.uniform(0.2,0.9)
    l = np.random.uniform(0.2,0.8)
    
    r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]

    tree = mat.node_tree
    nodes = tree.nodes
    nodes.new(type = 'ShaderNodeBsdfPrincipled')
    
    BSDF = nodes['Principled BSDF']
    BSDF.inputs['Base Color'].default_value = (r, g, b, 1)

    BSDF.inputs['Specular'].default_value = np.random.uniform(0.6,0.9)
    BSDF.inputs['Roughness'].default_value = np.random.uniform(0.1,0.25)
    inpt = nodes["Material Output"].inputs['Surface']
    output = BSDF.outputs[0]

    tree.links.new(inpt, output)

    return mat

def constrain_camera(cam, location=(0,0,0)):
    bpy.ops.object.empty_add()
    constraint_empty = bpy.data.objects['Empty']
    constraint_empty.name = 'camera_constraint_empty'
    constraint_empty.location = location
    
    bpy.ops.object.empty_add()
    parent_empty = bpy.data.objects['Empty']
    parent_empty.name = 'camera_parent_empty'
        
    cam.parent = parent_empty

    cam_constraint = parent_empty.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_constraint.target = constraint_empty
    
    return constraint_empty, parent_empty

def add_camera(cam_params):
    
    location, rotation = (0,0,0), (0,0,0)
    rotation = tuple(np.radians(rotation))

    bpy.ops.object.camera_add(
            enter_editmode=False, 
            align='VIEW', 
            location=location, 
            rotation=rotation, 
            scale=(1, 1, 1))

    camera = bpy.data.objects['Camera']
    camera.data.lens = cam_params['focal_length']
    camera.data.sensor_height = cam_params['sensor_height']
    camera.data.sensor_width = cam_params['sensor_width']

    return camera

def add_rigid_body_property(obj_list, p_type):

    for obj in obj_list:

        obj = bpy.data.objects[obj]
        bpy.context.view_layer.objects.active = None
        bpy.context.view_layer.objects.active = obj

        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.type = p_type
        bpy.context.object.rigid_body.mass = 0.1
        bpy.context.object.rigid_body.linear_damping = 0.7
        bpy.context.object.rigid_body.angular_damping = 0.8
        bpy.context.object.rigid_body.restitution = 0.7

        bpy.context.scene.rigidbody_world.use_split_impulse = True
        bpy.context.scene.rigidbody_world.time_scale = 10

        bpy.context.object.rigid_body.collision_shape = 'CONVEX_HULL'
        bpy.context.object.rigid_body.collision_margin = 0.001

def remove_rigid_body_property(obj_list):

    for obj in obj_list:
        obj = bpy.data.objects[obj]
        bpy.context.view_layer.objects.active = None
        bpy.context.view_layer.objects.active = obj

        bpy.ops.rigidbody.object_remove() 

def add_IBL(hdr_file, hdr_path, strength):
    
    hdr_img_path = os.path.join(hdr_path, hdr_file)
    bhdr_img_path = os.path.join(hdr_path, hdr_file.replace('.hdr', '_blurred.hdr'))
    
    world = bpy.context.scene.world
    world.use_nodes = True
    
    nodes = world.node_tree.nodes

    nodes.new(type="ShaderNodeTexEnvironment")
    hr_env_node = nodes['Environment Texture']
    hr_env_node.name = 'HR Texture'

    nodes.new(type="ShaderNodeTexEnvironment")
    b_env_node = nodes['Environment Texture']
    b_env_node.name = 'B Texture'

    nodes.new(type="ShaderNodeTexCoord")
    tex_co_node = nodes['Texture Coordinate']


    nodes.new(type="ShaderNodeMapping")
    map_node = nodes['Mapping']

    hr_node_bg = nodes['Background']
    hr_node_bg.name = 'HR Background'
    
    nodes.new(type="ShaderNodeBackground")
    b_node_bg = nodes['Background']
    b_node_bg.name = 'B Background'

    nodes.new(type="ShaderNodeMixShader")
    mix_node = nodes['Mix Shader']
    
    nodes.new(type="ShaderNodeLightPath")
    lp_node = nodes['Light Path']

    # connecting nodes    
    world.node_tree.links.new(
            tex_co_node.outputs['Generated'],
            map_node.inputs['Vector'])
    
    world.node_tree.links.new(
            map_node.outputs['Vector'], 
            hr_env_node.inputs['Vector'])

    world.node_tree.links.new(
            map_node.outputs['Vector'], 
            b_env_node.inputs['Vector'])
    
    world.node_tree.links.new(
            hr_env_node.outputs['Color'], 
            hr_node_bg.inputs['Color'])

    world.node_tree.links.new(
            b_env_node.outputs['Color'],
            b_node_bg.inputs['Color'])
    
    world.node_tree.links.new(
            hr_node_bg.outputs['Background'],
            mix_node.inputs[1])
    
    world.node_tree.links.new(
            b_node_bg.outputs['Background'],
            mix_node.inputs[2])
    
    world.node_tree.links.new(
            lp_node.outputs['Is Camera Ray'],
            mix_node.inputs['Fac'])
    
    world.node_tree.links.new(
            mix_node.outputs['Shader'], 
            nodes['World Output'].inputs['Surface'])


    bpy.ops.image.open(filepath=hdr_img_path)
    hr_bg_img = bpy.data.images[hdr_file]

    if os.path.isfile(bhdr_img_path):
        bpy.ops.image.open(filepath=bhdr_img_path)
        b_bg_img = bpy.data.images[hdr_file.replace('.hdr', '_blurred.hdr')]
    else:
        bpy.ops.image.open(filepath=hdr_img_path)
        b_bg_img = bpy.data.images[hdr_file]

    b_node_bg.inputs['Strength'].default_value = strength
    hr_node_bg.inputs['Strength'].default_value = strength
    
    hr_env_node.image = hr_bg_img
    b_env_node.image = b_bg_img

    return hdr_file, map_node


def add_PBR(name, pbr_file, pbr_path):
   
    pbr_path = os.path.join(pbr_path, pbr_file)
    
    img_path = os.path.join(pbr_path, next(x for x in os.listdir(pbr_path) if 'Color' in x))
    displ_path = os.path.join(pbr_path, next(x for x in os.listdir(pbr_path) if 'Displacement' in x))
    normal_path = os.path.join(pbr_path, next(x for x in os.listdir(pbr_path) if 'Normal' in x))
    rough_path = os.path.join(pbr_path, next(x for x in os.listdir(pbr_path) if 'Roughness' in x))
    
    bpy.data.materials.new(name)
    mat = bpy.data.materials[name]
    mat.use_nodes = True
   
    tree = mat.node_tree
    nodes = tree.nodes
    
    BSDF = nodes['Principled BSDF']
    inpt = nodes["Material Output"].inputs['Surface']
    output = BSDF.outputs[0]

    tree.links.new(inpt, output)
    
    #creating input nodes
    color = add_texture_node(nodes, "color")
    normal = add_texture_node(nodes, "normal")
    displacement = add_texture_node(nodes, "displacement")
    roughness = add_texture_node(nodes, "roughness")
    
    #loading images
    bpy.ops.image.open(filepath=img_path)
    color_img = next(x for x in bpy.data.images if 'Color' in x.name)
    
    bpy.ops.image.open(filepath=rough_path)
    rough_img = next(x for x in bpy.data.images if 'Roughness' in x.name)
    rough_img.colorspace_settings.name = 'Non-Color'

    bpy.ops.image.open(filepath=displ_path)
    displ_img = next(x for x in bpy.data.images if 'Displacement' in x.name)
    displ_img.colorspace_settings.name = 'Non-Color'
    
    bpy.ops.image.open(filepath=normal_path)
    normal_img = next(x for x in bpy.data.images if 'Normal' in x.name)
    normal_img.colorspace_settings.name = 'Non-Color'

    color.image = color_img
    roughness.image = rough_img
    displacement.image = displ_img
    normal.image = normal_img


    #connecting nodes
    color_output = color.outputs['Color']
    color_input = BSDF.inputs['Base Color']
    tree.links.new(color_input, color_output)
        
    # roughness
    roughness_output = roughness.outputs['Color']
    roughness_input = BSDF.inputs['Roughness']
    tree.links.new(roughness_input, roughness_output)

    # displacement
    dsp_vec_node = nodes.new(type='ShaderNodeDisplacement')
    displ_output = displacement.outputs['Color']
    dsp_vec_input = dsp_vec_node.inputs['Height']
    tree.links.new(dsp_vec_input, displ_output)

    dsp_vec_output = dsp_vec_node.outputs['Displacement']
    dsp_input = nodes["Material Output"].inputs['Displacement']
    tree.links.new(dsp_vec_output, dsp_input)

    # normal
    normal_vec_node = nodes.new(type='ShaderNodeNormalMap')
    normal_output = normal.outputs['Color']
    normal_vec_input = normal_vec_node.inputs['Color']
    tree.links.new(normal_output, normal_vec_input)

    normal_vec_output = normal_vec_node.outputs['Normal']
    normal_input = BSDF.inputs['Normal']
    tree.links.new(normal_vec_output, normal_input)


    # UV map node
    UV_node = nodes.new(type="ShaderNodeUVMap")
    mapping_node = nodes.new(type="ShaderNodeMapping")

    mapping_node.inputs["Scale"].default_value = (5.0,5.0,1.0)
    
    uv_node_output = UV_node.outputs['UV']
    mapping_node_input = mapping_node.inputs['Vector']
    tree.links.new(uv_node_output, mapping_node_input)

    mapping_node_output = mapping_node.outputs['Vector']

    tree.links.new(mapping_node_output, color.inputs['Vector'])
    tree.links.new(mapping_node_output, roughness.inputs['Vector'])
    tree.links.new(mapping_node_output, displacement.inputs['Vector'])
    tree.links.new(mapping_node_output, normal.inputs['Vector'])

    return mat.name

def analyze_scene(obj_list):
    scene = bpy.data.scenes['Scene']
    frameInfo = []
    
    import copy
    for frame in range(scene.frame_start,
                       scene.frame_end,
                       scene.frame_step):
        scene.frame_set(frame)
        
        frame_data = {}
        for obj in obj_list:

            obj = bpy.data.objects[obj]
            frame_data[obj.name] = copy.deepcopy(obj.matrix_world)
            frameInfo.append(frame_data)

    return frameInfo

def apply_pose(obj_list, frame_data):

    for obj in obj_list:

        obj = bpy.data.objects[obj]
        obj.matrix_world = frame_data[obj.name] 

def add_empty(co):
    bpy.ops.object.empty_add(
            type='PLAIN_AXES',
            radius=0.05,
            align='WORLD', 
            location=co, 
            scale=(0.05, 0.05, 0.05))

def do_compositing(path, obj_list):
    
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True
    bpy.context.scene.view_layers["View Layer"].use_pass_z = True
    bpy.context.scene.view_layers["View Layer"].use_pass_glossy_color = True
    bpy.context.scene.view_layers["View Layer"].use_pass_diffuse_color = True
    bpy.context.scene.view_layers["View Layer"].use_pass_transmission_color = True
    bpy.context.scene.view_layers["View Layer"].pass_alpha_threshold = 0


    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    
    nodes.new(type='CompositorNodeOutputFile')
    output_node = nodes['File Output']
    ###########################################################
    output_node.name = 'RGB File Output'
    output_node.file_slots[0].path = ''
    output_node.base_path = os.path.join(path, 'RGB')
    
    render_node = nodes['Render Layers']

    file_output_in = output_node.inputs[0]
    render_node_out = render_node.outputs['Image']

    tree.links.new(file_output_in, render_node_out)
    #################################################################
    
    ### segmentation output
    for i, obj in enumerate(obj_list):
        obj = bpy.data.objects[obj]
        obj.pass_index = i+1
        
        ## adding nodes
        nodes.new(type='CompositorNodeIDMask')
        node_name = obj.name + '_id_mask'
        nodes['ID Mask'].name = node_name
        print (list(nodes))
        id_mask_node = nodes[node_name]
        id_mask_node.index = i+1

        nodes.new(type='CompositorNodeMixRGB')
        node_name = obj.name + '_mix'
        nodes['Mix'].name = node_name
        mix_node = nodes[node_name]

        mix_node.inputs[1].default_value = (0, 0, 0, 1)
        mix_node.inputs[2].default_value = (1, 1, 1, 1)

        nodes.new(type='CompositorNodeOutputFile')
        node_name = obj.name + '_output'
        nodes['File Output'].name = node_name
        seg_output_node = nodes[node_name] 

        seg_output_node.file_slots[0].path = '' + obj.name + '_'
        seg_output_node.base_path = os.path.join(
                path, 
                'segmentations', 
                obj.name
            )

        ## linking nodes
        tree.links.new(render_node.outputs['IndexOB'], 
                       id_mask_node.inputs['ID value'])

        tree.links.new(id_mask_node.outputs['Alpha'],
                       mix_node.inputs['Fac'])

        tree.links.new(mix_node.outputs['Image'],
                       seg_output_node.inputs[0])

    #########################################################
    
    #### DEPTH #####

    ### depth output EXR
    nodes.new(type='CompositorNodeOutputFile')
    node_name = 'depth_exr_output'
    nodes['File Output'].name = node_name
    depth_exr_output_node = nodes[node_name]
    depth_exr_output_node.format.file_format = 'OPEN_EXR'
    depth_exr_output_node.format.color_depth = '32'
    depth_exr_output_node.format.color_mode = 'RGB'
    depth_exr_output_node.file_slots[0].path = ''
    depth_exr_output_node.base_path = os.path.join(path, 'depth_exr')

    nodes.new(type='CompositorNodeOutputFile')
    node_name = 'depth_png_output'
    nodes['File Output'].name = node_name
    depth_png_output_node = nodes[node_name] 
    depth_png_output_node.format.file_format = 'PNG'
    depth_png_output_node.format.color_depth = '16'
    depth_png_output_node.format.color_mode = 'BW'
    depth_png_output_node.file_slots[0].path = ''
    depth_png_output_node.base_path = os.path.join(path, 'depth_png')

    nodes.new(type='CompositorNodeNormalize')
    node_name = 'normalize'
    nodes['Normalize'].name = node_name
    normalize_node = nodes[node_name] 

    # linking render node to png output
    tree.links.new(render_node.outputs['Depth'],
                   normalize_node.inputs[0])

    tree.links.new(normalize_node.outputs[0],
                   depth_png_output_node.inputs[0])
    
    # linking render node to exr output
    tree.links.new(render_node.outputs['Depth'],
                   depth_exr_output_node.inputs[0])
    #################################################################
    """
    #### ALBEDO #####
    nodes.new(type='CompositorNodeOutputFile')
    node_name = 'diff_albedo_output'
    nodes['File Output'].name = node_name
    diff_albedo_output_node = nodes[node_name] 
    diff_albedo_output_node.format.file_format = 'PNG'
    diff_albedo_output_node.format.color_depth = '16'
    diff_albedo_output_node.format.color_mode = 'RGB'
    diff_albedo_output_node.file_slots[0].path = ''
    diff_albedo_output_node.base_path = os.path.join(path, 'diff_albedo')
    nodes.new(type='CompositorNodeOutputFile')
    node_name = 'gloss_albedo_output'
    nodes['File Output'].name = node_name
    gloss_albedo_output_node = nodes[node_name] 
    gloss_albedo_output_node.format.file_format = 'PNG'
    gloss_albedo_output_node.format.color_depth = '16'
    gloss_albedo_output_node.format.color_mode = 'RGB'
    gloss_albedo_output_node.file_slots[0].path = ''
    gloss_albedo_output_node.base_path = os.path.join(path, 'gloss_albedo')
    
    nodes.new(type='CompositorNodeOutputFile')
    node_name = 'trans_png_output'
    nodes['File Output'].name = node_name
    trans_albedo_output_node = nodes[node_name] 
    trans_albedo_output_node.format.file_format = 'PNG'
    trans_albedo_output_node.format.color_depth = '16'
    trans_albedo_output_node.format.color_mode = 'RGB'
    trans_albedo_output_node.file_slots[0].path = ''
    trans_albedo_output_node.base_path = os.path.join(path, 'trans_albedo')
    
    tree.links.new(render_node.outputs['DiffCol'],
                   diff_albedo_output_node.inputs[0])
    
    tree.links.new(render_node.outputs['TransCol'],
                   gloss_albedo_output_node.inputs[0])
    
    tree.links.new(render_node.outputs['GlossCol'],
                   trans_albedo_output_node.inputs[0])
    """
def remove_materials():
    # remove all materials in current project #
    materials = [mat.name for mat in bpy.data.materials]
    for material in materials:
        bpy.data.materials.remove(bpy.data.materials[material])
#
def add_lambertian_material(data):

    data.materials.new('lambertian')
    mat = bpy.data.materials['lambertian']
    mat.use_nodes = True

    return mat

def assign_material(obj, material_name):

    obj.data.materials.append(None)
    for slot in obj.material_slots:
        slot.material = bpy.data.materials[material_name].copy()

def add_checkered_material(data):
    
    name = "checkered"
    assert isinstance(name, str)

    data.materials.new(name)
    mat = bpy.data.materials[name]
    mat.use_nodes = True
   
    h = np.random.uniform(0,1)
    s = np.random.uniform(0.4,1)
    l = np.random.uniform(0.4,0.6)
    
    r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]

    tree = mat.node_tree
    nodes = tree.nodes
    nodes.new(type = 'ShaderNodeBsdfPrincipled')
    nodes.new(type = 'ShaderNodeTexChecker')
    
    BSDF = nodes['Principled BSDF']
    BSDF.inputs['Base Color'].default_value = (r, g, b, 1)

    BSDF.inputs['Specular'].default_value = 0
    BSDF.inputs['Roughness'].default_value = 0
    inpt = nodes["Material Output"].inputs['Surface']
    output = BSDF.outputs[0]

    tree.links.new(inpt, output)

    tex = nodes['Checker Texture']
    top_end = np.random.uniform(0.75, 1.0)
    low_end = np.random.uniform(0.0, 0.25)

    tex.inputs['Color1'].default_value = (low_end, low_end, low_end, 1)
    tex.inputs['Color2'].default_value = (top_end, top_end, top_end,1)
    tex.inputs['Scale'].default_value = np.random.choice(np.arange(3,6))
    
    inpt=tex.outputs['Color']
    output=BSDF.inputs['Base Color']

    tree.links.new(inpt, output)

    return mat

def add_voronoi_material(data):
    
    name = "voronoi"
    assert isinstance(name, str)

    data.materials.new(name)
    mat = bpy.data.materials[name]
    mat.use_nodes = True
   
    h = np.random.uniform(0,1)
    s = np.random.uniform(0.4,1)
    l = np.random.uniform(0.4,0.6)
    
    r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]

    tree = mat.node_tree
    nodes = tree.nodes
    nodes.new(type = 'ShaderNodeBsdfPrincipled')
    nodes.new(type = 'ShaderNodeTexVoronoi')
    
    BSDF = nodes['Principled BSDF']
    BSDF.inputs['Base Color'].default_value = (r, g, b, 1)

    BSDF.inputs['Specular'].default_value = 0
    BSDF.inputs['Roughness'].default_value = 0
    inpt = nodes["Material Output"].inputs['Surface']
    output = BSDF.outputs[0]

    tree.links.new(inpt, output)

    tex = nodes['Voronoi Texture']

    tex.inputs['Randomness'].default_value = np.random.uniform(0,1)
    tex.inputs['Scale'].default_value = np.random.choice(np.arange(3,6))
    
    inpt=tex.outputs['Color']
    output=BSDF.inputs['Base Color']

    tree.links.new(inpt, output)

    return mat
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

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

def apply_settings(render_params, scn):
    scn.render.engine = 'CYCLES'
    scn.cycles.device = 'GPU'
    scn.render.tile_x = render_params['render_tile_x']
    scn.render.tile_y = render_params['render_tile_y']
    
    scn.render.resolution_x = render_params['resolution_x']
    scn.render.resolution_y = render_params['resolution_y']
    scn.render.use_persistent_data = True
    scn.render.use_save_buffers = True

    scn.cycles.samples = render_params['samples']
    scn.cycles.use_denoising = render_params['use_denoising']

    scn.cycles.debug_use_spatial_splits = render_params['use_spatial_splits']

    #scn.cycles.max_bounces = render_params['min_bouncs']
    #scn.cycles.min_bounces = render_params['max_bounces']

    #scn.cycles.caustics_refractive = render_params['use_caustics_refractive']
    #scn.cycles.caustics_reflective = render_params['use_caustics_reflective']


def add_clvr_object(object_dir, name, scale, loc, theta=0):
  """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.
  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  """
  # First figure out how many of this object are already in the scene so we can
  # give the new object a unique name
  count = 0
  for obj in bpy.data.objects:
    if obj.name.startswith(name):
      count += 1

  filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
  bpy.ops.wm.append(filename=filename)

  # Give it a new name to avoid conflicts
  new_name = '%s_%d' % (name, count)
  bpy.data.objects[name].name = new_name

  # Set the new object as active, then rotate, scale, and translate it
  x, y = loc
  obj = bpy.data.objects[new_name]
  bpy.context.view_layer.objects.active = obj
  obj.select_set(True)
  # bpy.context.scene.objects.active = bpy.data.objects[new_name]
  bpy.context.object.rotation_euler[2] = theta
  bpy.ops.transform.resize(value=(scale, scale, scale))
  bpy.ops.transform.translate(value=(x, y, scale))

  return obj


def add_toys4k_obj(scn, path, category, name, scale, loc, theta=0):

    filename = os.path.join(path)

    with bpy.data.libraries.load(filename, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects


    print (data_to.objects[0].name)
    name = 'object'
    data_to.objects[0].name = name


    for obj in data_to.objects:
        if obj is not None and type(obj) != 'str':
            scn.collection.objects.link(obj)


    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc 
    print (bpy.data.objects[name], name)
    obj = bpy.data.objects[name] 
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # bpy.context.scene.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, scale))

    return obj


def load_clvr_materials(material_dir):
  """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'): continue
    name = os.path.splitext(fn)[0]
    filepath = os.path.join(material_dir, fn, 'NodeTree', name)
    bpy.ops.wm.append(filename=filepath)


def add_clvr_material(name, **properties):
  """
  Create a new material and assign it to the active object. "name" should be the
  name of a material that has been previously loaded using load_materials.
  """
  # Figure out how many materials are already in the scene
  mat_count = len(bpy.data.materials)

  # Create a new material; it is not attached to anything and
  # it will be called "Material"
  bpy.ops.material.new()

  # Get a reference to the material we just created and rename it;
  # then the next time we make a new material it will still be called
  # "Material" and we will still be able to look it up by name
  mat = bpy.data.materials['Material']
  mat.name = 'Material_%d' % mat_count

  # Attach the new material to the active object
  # Make sure it doesn't already have materials
  obj = bpy.context.active_object
  assert len(obj.data.materials) == 0
  obj.data.materials.append(mat)

  # Find the output node of the new material
  output_node = None
  for n in mat.node_tree.nodes:
    if n.name == 'Material Output':
      output_node = n
      break

  # Add a new GroupNode to the node tree of the active material,
  # and copy the node tree from the preloaded node group to the
  # new group node. This copying seems to happen by-value, so
  # we can create multiple materials of the same type without them
  # clobbering each other
  group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
  group_node.node_tree = bpy.data.node_groups[name]

  # Find and set the "Color" input of the new group node
  for inp in group_node.inputs:
    if inp.name in properties:
      inp.default_value = properties[inp.name]

  # Wire the output of the new group node to the input of
  # the MaterialOutput node
  mat.node_tree.links.new(
      group_node.outputs['Shader'],
      output_node.inputs['Surface'],
  )


