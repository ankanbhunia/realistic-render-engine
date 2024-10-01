from generate import *
import glob, random
data_path = "/home/s2514643/MOAD-data-generation-latest/toys4k_blend_files/"
out_path = "/home/s2514643/MOAD-data-generation-latest/output_data/raw_blend_files/"
total_iterations = 500
max_create = 4

catergories = ['orange', 'fox', 'horse', 'mouse', 'hamburger', 'bottle', 'hat', 'key', 
'butterfly', 'mug', 'giraffe', 'frog', 'bunny', 'chairs', 'plate', 'fish', 'shark', 'cow',
 'ice_cream', 'bowl', 'bread', 'chess_piece', 'hammer', 'light_bulb', 'elephant', 'boat', 'car',
  'deer_moose', 'octopus', 'airplane', 'sheep', 'banana', 'train', 'cake', 'pig', 'bus', 'chicken', 
  'robot', 'apple', 'cookie', 'glass', 'cupcake', 'lion', 'donut', 'dinosaur', 'panda', 'dog', 'monkey', 
  'lizard', 'shoe', 'penguin', 'cup']


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

    run_function(agp.BumpAnomalyGenerator, blend_file_path, output_folder, model_name, \
                mode = 'bump')