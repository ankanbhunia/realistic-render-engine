import json
import os

def main():
    dirpath = "/home/sstojanov3/hdd_data/modelnet/modelnet40_aligned_obj"
    
    dct = {}
    categs = [x for x in sorted(os.listdir(dirpath))]

    for categ in categs:
        categ_path = os.path.join(dirpath, categ)
        train_objects = sorted(os.listdir(os.path.join(categ_path, "train")))
        test_objects = sorted(os.listdir(os.path.join(categ_path, "test")))
        
        train_objects = [os.path.join("train", x) for x in train_objects]
        test_objects = [os.path.join("test", x) for x in test_objects]

        dct[categ] = train_objects + test_objects

    out_str = json.dumps(dct, indent=True)

    with open("modelnet_dict.json", "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    main()
