from generate import *
import glob, random
from anomaly_generators.utils import *
import tqdm 

out_path = "/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files/"
data_path = "/home/s2514643/MOAD-data-generation-latest/toys4k_blend_files/"


samples = [i+i.split('/')[-2]+'.blend' for i in  glob.glob(f'{out_path}/*/')]

for idx, blend_file_path in enumerate(samples):

    print(f"#### Starting iter {idx}")

    if not os.path.isfile(blend_file_path):

        v = blend_file_path.split('/')
        print ('copying file..')
        shutil.copyfile(data_path+'/'+'_'.join(v[-2].split('_')[:-1])+'/'+v[-2]+'/'+v[-1], blend_file_path)
        
    if not os.path.isfile(blend_file_path.replace('.blend', '.ply')):

        load_blend_file(blend_file_path)
        original_mesh = load_scene_as_obj()
        pnts = trimesh.sample.sample_surface_even(original_mesh, 20000, radius = 0.04)[0]
        trimesh.PointCloud(pnts).export(blend_file_path.replace('.blend', '.ply'))
