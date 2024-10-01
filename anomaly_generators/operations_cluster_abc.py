import sys
sys.path.append('/home/s2514643/MOAD-data-generation-latest/MOAD-v1')
import json
import trimesh
import tqdm, glob
from anomaly_generators.utils import *

def scale_mesh(mesh1, mesh2):
    # Calculate bounding boxes
    bbox1 = mesh1.bounds
    bbox2 = mesh2.bounds
    
    # Calculate scaling factor
    scaling_factor = (bbox1.ptp(axis=0) / bbox2.ptp(axis=0)).mean()
    
    # Apply scaling factor to mesh2
    scaled_mesh2 = mesh2.apply_scale(scaling_factor)
    
    return scaled_mesh2

data_path = '/home/s2514643/MOAD-data-generation-latest/abc_dataset/'

with open("/disk/scratch/s2514643/topk_indices.json", "r") as f:
    cluster = json.load(f)

map_dict = {}
for path, indices in zip(cluster['paths'], cluster['topk_indices']):
    map_dict[path.split('/')[-2]] = [cluster['paths'][idx].split('/')[-2] for idx in indices]
    

query_id = '00006777_edfc854d229f4870ab72cb02_trimesh_039'

def get_close_shapes(query_id, thres = 5, num = 2):
    path1 = data_path +'/'+ query_id + '.obj'
    mesh1 = trimesh.load_mesh(path1)
    verts_list = [mesh1.vertices.shape[0]]
    out_meshs = []
    c = 0
    for i in range(1,50):
        if c == num:
            break
        path2 = data_path +'/'+ map_dict[query_id][i] + '.obj'
        mesh2 = trimesh.load_mesh(path2)
        del_p = [abs(mesh2.vertices.shape[0]-verts_list[j])/verts_list[j]*100 for j in range(len(verts_list))]
        if min(del_p)>thres:
            out_mesh = scale_mesh(mesh1, mesh2)
            out_meshs.append(out_mesh)
            c = c + 1
            verts_list.append(mesh2.vertices.shape[0])
        print (del_p)
    return mesh1, out_meshs
    

data_path = '/home/s2514643/MOAD-data-generation-latest/abc_dataset/'
base_path = '/home/s2514643/MOAD-data-generation-latest/abc_dataset/'
out_folder = '/home/s2514643/MOAD-data-generation-latest/abc_anomaly/'

existing_objs = glob.glob(out_folder+'/*')

mode = 'diff'

for existing_obj in tqdm.tqdm(existing_objs):

    query_id = existing_obj.split('/')[-1]

    mesh, out_meshes = get_close_shapes(query_id, thres = 5, num = 2)

    for mesh_p in out_meshes:

        model_path = out_folder+'/'+query_id
        output_file_path = get_file_path_obj(model_path, query_id, mode).replace('.blend', '.obj')  
        mesh_p.export(output_file_path)

    
