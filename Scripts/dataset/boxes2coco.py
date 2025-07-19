import os
import json
import shutil
from tqdm import tqdm
import pathlib
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Extract coco format object detection annotations from 3RScan dataset.')
parser.add_argument('--path', type=pathlib.Path, required=True, help='3RScan directory')
parser.add_argument('--label_categories', type=str, choices=['scannet', 'replica'], default='scannet', help='Label categories to use')
args = parser.parse_args()
args.path = args.path.resolve()

# Define paths
if args.label_categories == 'scannet':
    object_json_path = args.path / "3DSSG/objects.json"
    categories_path = args.path / "3DSSG_subset/classes.txt"
    train_scans = args.path / "3DSSG_subset/train_scans.txt"
    valid_scans = args.path / "3DSSG_subset/validation_scans.txt"
    test_scans = args.path / "3DSSG_subset/test_scans.txt"
    mapping_path = args.path / "3DSSG_subset/3dssg_to_scannet.json"
    rel_json_path = args.path / "3DSSG_subset/relationships.json"
    data_folder = args.path / "data"
    obj_output_folder = str(args.path / "ObjectDetection20")
    rel_output_folder = str(args.path / "2DSG20")
elif args.label_categories == 'replica':
    object_json_path = args.path / "ReplicaSSG/objects.json"
    train_scans = args.path / "ReplicaSSG/train_scans.txt"
    valid_scans = args.path / "ReplicaSSG/validation_scans.txt"
    test_scans = args.path / "ReplicaSSG/test_scans.txt"
    mapping_path = args.path / "ReplicaSSG/replica_to_visual_genome.json"
    rel_json_path = args.path / "ReplicaSSG/relationships.json"
    data_folder = args.path / "data"
    obj_output_folder = str(args.path / "ObjectDetection")
    rel_output_folder = str(args.path / "2DSG")

# Create output folders
os.makedirs(f"{obj_output_folder}/images/train", exist_ok=True)
os.makedirs(f"{obj_output_folder}/images/val", exist_ok=True)
os.makedirs(f"{obj_output_folder}/images/test", exist_ok=True)
os.makedirs(f"{obj_output_folder}/coco", exist_ok=True)
os.makedirs(f"{rel_output_folder}", exist_ok=True)

# Load categories
with open(mapping_path, 'r') as f:
    mapping = json.load(f)
if args.label_categories == 'scannet':
    categories = mapping['ScanNet_list']
elif args.label_categories == 'replica':
    categories = mapping['VisualGenome_list']

# Create coco format
coco_train = {"images": [], "annotations": [], "categories": []}
coco_valid = {"images": [], "annotations": [], "categories": []}
coco_test = {"images": [], "annotations": [], "categories": []}
categories2ID = {}
for i, category in enumerate(categories):
    coco_train["categories"].append({"id": i+1, "name": category})
    coco_valid["categories"].append({"id": i+1, "name": category})
    coco_test["categories"].append({"id": i+1, "name": category})
    categories2ID[category] = i+1

# Load scan splits
with open(train_scans, 'r') as f:
    train_scan_names = f.readlines()
train_scan_names = [scan.strip() for scan in train_scan_names]

with open(valid_scans, 'r') as f:
    valid_scan_names = f.readlines()
valid_scan_names = [scan.strip() for scan in valid_scan_names]

with open(test_scans, 'r') as f:
    test_scan_names = f.readlines()
test_scan_names = [scan.strip() for scan in test_scan_names]

# Load object data
with open(object_json_path, 'r') as f:
    object_data = json.load(f)
OBJ2CLASS = {}
for scan in object_data["scans"]:
    new_scan = scan["scan"]
    id_to_label = {}
    for obj in scan["objects"]:
        if args.label_categories == 'scannet':
            if mapping['3DSSG2NYUv2'][obj["label"]] not in categories2ID:
                id_to_label[obj["id"]] = -1
            else:
                id_to_label[obj["id"]] = categories2ID[mapping['3DSSG2NYUv2'][obj["label"]]]
        elif args.label_categories == 'replica':
            if obj["label"] not in mapping["Replica2VisualGenome"] or mapping['Replica2VisualGenome'][obj["label"]] not in categories2ID:
                id_to_label[obj["id"]] = -1
            else:
                id_to_label[obj["id"]] = categories2ID[mapping['Replica2VisualGenome'][obj["label"]]]
    OBJ2CLASS[new_scan] = id_to_label

# Load relationships
with open(rel_json_path, 'r') as f:
    rel_data = json.load(f)
SCAN2REL = {}
for scan in rel_data['scans']:
    SCAN2REL[scan['scan']] = scan['relationships']

# Create rel format
rel = {"train": {}, "val": {}, "test": {}, "rel_categories": []}
if args.label_categories == 'scannet':
    rel['rel_categories'] = mapping['ScanNet_rel']
elif args.label_categories == 'replica':
    rel['rel_categories'] = mapping['VisualGenome_rel']


data_dirs = sorted([dir for dir in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, dir))])
cur_img_id = -1
cur_ann_id = 0
not_in_scan = 0

for dir in tqdm(data_dirs):
    images = sorted([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("color.jpg") and not f.endswith("rendered.color.jpg")])
    boxes = sorted([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("bb.txt")])
    if args.label_categories == 'scannet':
        visibilities = sorted([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("visibility.txt")])
        assert len(images) == len(boxes) == len(visibilities), f"Number of images ({len(images)}), boxes ({len(boxes)}) and visibilities ({len(visibilities)}) do not match for {dir}"
    elif args.label_categories == 'replica':
        assert len(images) == len(boxes), f"Number of images ({len(images)}) and boxes ({len(boxes)}) do not match for {dir}"

    if dir in train_scan_names:
        coco = coco_train
        split = "train"
    elif dir in valid_scan_names:
        coco = coco_valid
        split = "val"
    elif dir in test_scan_names:
        coco = coco_test
        split = "test"
    else:
        not_in_scan += 1
        continue

    with open(os.path.join(data_folder, dir, "sequence", "_info.txt"), 'r') as f:
        for line in f.readlines():
            if line.startswith("m_colorWidth"):
                seq_width = int(line.split(" = ")[1])
            if line.startswith("m_colorHeight"):
                seq_height = int(line.split(" = ")[1])

    scan_rels = SCAN2REL[dir]

    for i in range(len(images)):
        image, bb = images[i], boxes[i]
        if args.label_categories == 'scannet':
            vis = visibilities[i]
        cur_img_id += 1 # Prevent duplicate image ids
        img_idx = image[6:-10]
        if args.label_categories == 'scannet': # rotate images for scannet
            img_arr = cv2.imread(os.path.join(data_folder, dir, "sequence", image))
            img_arr = np.rot90(img_arr, 3)
            cv2.imwrite(f"{obj_output_folder}/images/{split}/{dir}-{img_idx}.jpg", img_arr)
        elif args.label_categories == 'replica':
            shutil.copyfile(os.path.join(data_folder, dir, "sequence", image), f"{obj_output_folder}/images/{split}/{dir}-{img_idx}.jpg")
        coco["images"].append({"id": cur_img_id, "file_name": f"{dir}-{img_idx}.jpg", "width": seq_width, "height": seq_height})
        rel[split][str(cur_img_id)] = []
        
        with open(os.path.join(data_folder, dir, "sequence", bb), 'r') as f:
            objects = [obj.split(" ") for obj in f.readlines()]
        if args.label_categories == 'scannet':
            with open(os.path.join(data_folder, dir, "sequence", vis), 'r') as f:
                vis_values = [list(map(float, visibility.split(" "))) for visibility in f.readlines()]
            if len(vis_values) == 0:
                continue
        obj_id_in_img = [] # global idx in an image
        OBJ_ID2IDX = {} # obj id in an image to global idx
        obj_id = 0 # obj id in an image
        for idx, obj in enumerate(objects):
            if args.label_categories == 'scannet' and vis_values[idx][3] * vis_values[idx][6] < 0.1:
                continue
            category_id = OBJ2CLASS[dir][obj[0]]
            if category_id == -1:
                continue
            x, y = int(obj[1]), int(obj[2])
            w, h = int(obj[3]) - x, int(obj[4]) - y
            if w*h < 1:
                continue
            if args.label_categories == 'scannet':
                bbox = [seq_height-y-h, x, h, w]
            elif args.label_categories == 'replica':
                bbox = [x, y, w, h]
            coco["annotations"].append({"id": cur_ann_id, "category_id": category_id, "iscrowd": 0, "image_id": cur_img_id, "area": w*h, "bbox": bbox})
            obj_id_in_img.append(int(obj[0]))
            OBJ_ID2IDX[int(obj[0])] = obj_id
            cur_ann_id += 1
            obj_id += 1
        
        for scan_rel in scan_rels:
            if scan_rel[0] in obj_id_in_img and scan_rel[1] in obj_id_in_img:
                if args.label_categories == 'scannet' and scan_rel[3] not in mapping['ScanNet_rel']:
                    continue
                if args.label_categories == 'replica' and scan_rel[3] not in mapping['VisualGenome_rel']:
                    continue
                if args.label_categories == 'scannet':
                    rel_id = mapping['ScanNet_rel'].index(scan_rel[3])
                elif args.label_categories == 'replica':
                    rel_id = mapping['VisualGenome_rel'].index(scan_rel[3])
                rel[split][str(cur_img_id)].append([OBJ_ID2IDX[scan_rel[0]], OBJ_ID2IDX[scan_rel[1]], rel_id])

# Write to obj json
with open(f"{obj_output_folder}/coco/train.json", 'w') as f:
    json.dump(coco_train, f)
with open(f"{obj_output_folder}/coco/val.json", 'w') as f:
    json.dump(coco_valid, f)
with open(f"{obj_output_folder}/coco/test.json", 'w') as f:
    json.dump(coco_test, f)

# Create symlinks for rel json
if not os.path.exists(f"{rel_output_folder}/train.json"):
    os.symlink(f"{obj_output_folder}/coco/train.json", f"{rel_output_folder}/train.json")
if not os.path.exists(f"{rel_output_folder}/val.json"):
    os.symlink(f"{obj_output_folder}/coco/val.json", f"{rel_output_folder}/val.json")
if not os.path.exists(f"{rel_output_folder}/test.json"):
    os.symlink(f"{obj_output_folder}/coco/test.json", f"{rel_output_folder}/test.json")
if not os.path.exists(f"{rel_output_folder}/images"):
    os.symlink(f"{obj_output_folder}/images", f"{rel_output_folder}/images")

# Write rel json
with open(f"{rel_output_folder}/rel.json", 'w') as f:
    json.dump(rel, f)