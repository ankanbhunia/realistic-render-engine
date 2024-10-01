import json
import os

def main():
    dirpath = "/home/sstojanov3/hdd_data/TOYS4K_BLEND_FILES_PACKED_V1"
    
    dct = {}
    categs = sorted(os.listdir(dirpath))

    for categ in categs:
        categ_path = os.path.join(dirpath, categ)
        objects = sorted(os.listdir(categ_path))
        objects = [os.path.join(x, '{}.blend'.format(x)) for x in objects]
        dct[categ] = objects
    
    out_str = json.dumps(dct, indent=True)

    with open("toys_dict.json", "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    main()
