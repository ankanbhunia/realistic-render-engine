
data_directory_path="/disk/scratch/s2514643/toys4k_blend_files/"
path_raw_blend_files="data/output_data/raw_blend_files"
categories=("apple" "ball" "banana" "bottle" "bowl" "bread" "cake" "candy" "cat" "chicken" "coin" "cookie" "cow" "cup" "cupcake" "dinosaur" "dog" "donut" "elephant" "fish" "flower" "frog" "grapes" "hat" "giraffe" "helmet" "panda" "orange" "penguin" "shoe" "mug" "shark" "whale")
#($(ls "$data_directory_path"))

for cat_id in "${categories[@]}"
do
    python create_anomaly.py --anomaly_type=fracture --category $cat_id --total_iterations=50 --out_path $path_raw_blend_files
    python create_anomaly.py --anomaly_type=bump --category $cat_id --total_iterations=50 --out_path $path_raw_blend_files
    python create_anomaly.py --anomaly_type=deform --category $cat_id --total_iterations=25 --out_path  $path_raw_blend_files
    python create_anomaly.py --anomaly_type=bend/twist --category $cat_id --total_iterations=25 --out_path $path_raw_blend_files
    python create_anomaly.py --anomaly_type=missing --category $cat_id --total_iterations=50 --out_path $path_raw_blend_files
    python create_anomaly.py --anomaly_type=3dTransform --category $cat_id --total_iterations=50 --out_path  $path_raw_blend_files
    python create_anomaly.py --anomaly_type=material --category $cat_id --total_iterations=50 --out_path  $path_raw_blend_files
done


