import json
from pycocotools.coco import COCO
from pathlib import Path
import os
import random
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot bounding boxes of 3RScan dataset.')
parser.add_argument('--path', type=Path, required=True, help='3RScan directory to check integrity')
parser.add_argument('--label_categories', type=str, choices=['scannet', 'replica'], default='scannet', help='Label categories to use')
args = parser.parse_args()

dataset_path = args.path
obj_file = dataset_path / f"2DSG{'20' if args.label_categories == 'scannet' else ''}/test.json"
rel_file = dataset_path / f"2DSG{'20' if args.label_categories == 'scannet' else ''}/rel.json"
os.makedirs("visualization/sg", exist_ok=True)

coco = COCO(obj_file)
colors = np.random.uniform(0, 255, size=(len(coco.getCatIds()), 3))

img_ids = coco.getImgIds()
imgs = coco.loadImgs(img_ids)
imgs = random.sample(imgs, 10)

with open(rel_file, "r") as f:
    rels = json.load(f)

classes = {"obj": [cat['name'] for cat in coco.loadCats(coco.getCatIds())], "rel": rels["rel_categories"]}

for img in imgs:
    I = cv2.imread(obj_file.parent / "images" / "test" / img['file_name'])
    ann_ids = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(ann_ids)
    print(f"Image: {img['file_name']}")
    for ann in anns:
        bbox = ann['bbox']
        x, y, w, h = bbox
        cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), colors[ann["category_id"]-1], 2)
        cv2.putText(I, coco.cats[ann['category_id']]["name"], (int(x+10), int(y+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    for rel in rels["test"][str(img['id'])]:
        cv2.arrowedLine(I, (int(anns[rel[0]]['bbox'][0] + anns[rel[0]]['bbox'][2] / 2), int(anns[rel[0]]['bbox'][1] + anns[rel[0]]['bbox'][3] / 2)), (int(anns[rel[1]]['bbox'][0] + anns[rel[1]]['bbox'][2] / 2), int(anns[rel[1]]['bbox'][1] + anns[rel[1]]['bbox'][3] / 2)), (0, 0, 255), 2)
        cv2.putText(I, classes["rel"][rel[2]], (int((anns[rel[0]]['bbox'][0] + anns[rel[0]]['bbox'][2] / 2 + anns[rel[1]]['bbox'][0] + anns[rel[1]]['bbox'][2] / 2) / 2), int((anns[rel[0]]['bbox'][1] + anns[rel[0]]['bbox'][3] / 2 + anns[rel[1]]['bbox'][1] + anns[rel[1]]['bbox'][3] / 2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        print(classes["obj"][anns[rel[0]]['category_id']-1], classes["rel"][rel[2]], classes["obj"][anns[rel[1]]['category_id']-1])

    cv2.imwrite(os.path.join("visualization/sg", img['file_name']), I)

