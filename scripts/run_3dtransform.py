from generate import *
import glob, random
data_path = "/home/s2514643/MOAD-data-generation-latest/toys4k_blend_files/"
out_path = "/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files/"
total_iterations = 500
max_create = 4

catergories = ["bottle", "airplane", "boat", "bunny", "bus", "cake", "car", "chairs", "frog", 
"chicken", "cow", "cup", "cupcake", "deer_moose", "dinosaur", "dog" , "hamburger", "elephant", "fish", "fox", 
"giraffe",  "horse", "ice_cream", "lion", "monkey", "mouse", "mug", 
"panda", "penguin", "pig", "robot", "train"]

all_paths = glob.glob(f'{data_path}/*/*/*')
samples = [i for i in all_paths if i.split('/')[-3] in catergories]

for idx in range(total_iterations):

    print(f"#### Starting iter {idx}")

    blend_file_path = random.choice(samples)
    model_name = blend_file_path.split('/')[-2]
    output_folder = f"{out_path}/{model_name}/"

    #if os.path.isdir(output_folder):
    #    if len(glob.glob(f"{output_folder}/*.blend"))>max_create:
    #        print (f'skipping... max_create is {max_create}')
    #        continue

    run_function(agp.Transform3DAnomalyGenerator, blend_file_path, output_folder, model_name, \
                min_region = 0.01, max_region = 0.5, num_iters = 500,  \
                min_trans = 0.1, max_trans = 0.4, min_angle =  0.3, max_angle = 0.4, \
                min_scale = 0.7, max_scale = 0.8, max_try = 10, \
                modes = ['tran', 'rota', 'scal'])