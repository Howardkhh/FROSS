import cv2
import numpy as np
import os
import random
import pathlib
import argparse
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(description='Plot bounding boxes of 3RScan dataset.')
parser.add_argument('--path', type=pathlib.Path, required=True, help='3RScan directory to check integrity')
parser.add_argument('--label_categories', type=str, choices=['scannet', 'replica'], default='scannet', help='Label categories to use')
args = parser.parse_args()

dataset_path = args.path
os.makedirs("visualization/bbox", exist_ok=True)

ann_file = dataset_path / f"ObjectDetection{'20' if args.label_categories == 'scannet' else ''}/coco/test.json"
coco = COCO(ann_file)
cat_ids = coco.getCatIds()
colors = np.random.uniform(0, 255, size=(len(coco.getCatIds()), 3))
img_ids = coco.getImgIds()
imgs = coco.loadImgs(img_ids)
imgs = random.sample(imgs, 10)
img_file_names = [img['file_name'] for img in imgs]

for img in imgs:
    ann_ids = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(ann_ids)
    img_path = os.path.join(dataset_path, f"ObjectDetection{'20' if args.label_categories == 'scannet' else ''}/images/test", img['file_name'])
    image = cv2.imread(img_path)
    for ann in anns:
        bbox = ann['bbox']
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[ann["category_id"]-1], 2)
        cv2.putText(image, coco.cats[ann['category_id']]["name"], (int(x1+10), int(y1+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join("visualization/bbox", img['file_name']), image)

