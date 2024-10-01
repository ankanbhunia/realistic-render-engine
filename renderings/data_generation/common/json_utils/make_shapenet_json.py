import json
import os

def main():
    dirpath = "/home/sstojanov3/hdd_data/ShapeNet_meshes/ShapeNetCore.v2"
    
    dct = {}
    categs = [x for x in sorted(os.listdir(dirpath)) if x.isnumeric()]

    for categ in categs:
        categ_path = os.path.join(dirpath, categ)
        objects = sorted(os.listdir(categ_path))

        objects = [os.path.join(x, 'models', 'model_normalized.obj') for x in objects]
        dct[categ] = objects
    out_str = json.dumps(dct, indent=True)

    with open("shapenet_dict.json", "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    main()
