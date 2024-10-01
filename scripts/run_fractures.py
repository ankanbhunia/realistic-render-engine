from generate import *
import glob, random
data_path = "/home/s2514643/MOAD-data-generation-latest/toys4k_blend_files/"
out_path = "/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files/"
total_iterations = 1000
max_create = 4

catergories = ["apple", "banana", "bottle", "bowl", "bread", "bunny", "butterfly", "cake", "car", "chairs", "chess_piece", 
"chicken", "cookie", "cow", "cup", "cupcake", "deer_moose", "dinosaur", "dog" , "donut", "elephant", "fish", "fox", "glass", 
"giraffe", "hammer", "hat", "horse", "ice_cream", "key", "light_bulb", "lion", "lizard", "monkey", "mouse", "mug", "octopus", "orange", 
"panda", "penguin", "pig", "plate", "robot", "shark", "sheep", "shoe", "train"]

all_paths = glob.glob(f'{data_path}/*/*/*')
samples = [i for i in all_paths if i.split('/')[-3] in catergories]

for idx in range(total_iterations):

    print(f"#### Starting iter {idx}")

    blend_file_path = random.choice(samples)
    model_name = blend_file_path.split('/')[-2]
    output_folder = f"{out_path}/{model_name}/"

    if os.path.isdir(output_folder):
        if len(glob.glob(f"{output_folder}/*.blend"))>max_create:
            print (f'skipping... max_create is {max_create}')
            continue

    run_function(agp.FractureAnomalyGenerator, blend_file_path, output_folder, model_name, \
        min_region = 0.02, max_region = 1.0, modes = ['frac', 'crack', 'disc'], num_iters = 500)