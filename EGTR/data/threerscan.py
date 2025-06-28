# References:
# - https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
# - https://github.com/suprosanna/relationformer/blob/scene_graph/datasets/get_dataset_counts.py
# - https://github.com/naver-ai/egtr/blob/7f87450f32758ed8583948847a8186f2ee8b21e3/data/visual_genome.py

import json
import os

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from PIL import Image


class ThreeRScanDetection(torchvision.datasets.CocoDetection):
    def __init__(self, data_folder, feature_extractor, split, debug=False):
        ann_file = os.path.join(data_folder, f"{split}.json")
        img_folder = os.path.join(data_folder, "images")
        super(ThreeRScanDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor
        self.split = split
        self.debug = debug
    
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, self.split, path)).convert("RGB")
    
    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(ThreeRScanDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
        target["class_labels"] -= 1  # remove 'no_relation' category
        return pixel_values, target

    def __len__(self):
        if self.debug and self.split == "train":
            return 5000
        else:
            return len(self.ids)


class ThreeRScanDataset(ThreeRScanDetection):
    def __init__(
        self, data_folder, feature_extractor, split, num_object_queries=100, debug=False
    ):
        super(ThreeRScanDataset, self).__init__(data_folder, feature_extractor, split, debug)
        with open(f"{data_folder}/rel.json", "r") as f:
            rel = json.load(f)
        self.rel = rel[split]
        self.rel_categories = rel["rel_categories"]
        self.num_object_queries = num_object_queries

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(ThreeRScanDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        rel_list = self.rel[str(image_id)]
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
        rel = np.array(rel_list)
        target["rel"] = self._get_rel_tensor(rel)
        target["class_labels"] -= 1
        return pixel_values, target

    def _get_rel_tensor(self, rel_tensor):
        indices = rel_tensor.T
        # indices[-1, :] -= 1  # there is no 'no_relation' category

        rel = torch.zeros([self.num_object_queries, self.num_object_queries, len(self.rel_categories)])
        if indices.shape[0] == 0:
            return rel
        rel[indices[0, :], indices[1, :], indices[2, :]] = 1.0
        return rel


# https://github.com/suprosanna/relationformer/blob/75c24f61a81466df8f40c498e5f7aae3edd5ac6b/datasets/get_dataset_counts.py#L9
def threerscan_get_statistics(train_data, must_overlap=True):
    """
    Get counts of all of the relations. Used for modeling directly P(rel | o1, o2)
    :param train_data:
    :param must_overlap:
    :return:
    """
    num_classes = len(train_data.coco.cats)
    num_predicates = len(train_data.rel_categories)

    fg_matrix = np.zeros(
        (
            num_classes + 1,
            num_classes + 1,
            num_predicates,
        ),
        dtype=np.int64,
    )

    rel = train_data.rel
    for idx in tqdm(range(len(train_data))):
        image_id = train_data.ids[idx]

        target = train_data.coco.loadAnns(train_data.coco.getAnnIds(image_id))
        gt_classes = np.array(list(map(lambda x: x["category_id"], target)))
        rel_list = rel[str(image_id)]
        if len(rel_list) == 0:
            continue
        gt_indices = np.array(torch.Tensor(rel_list).T, dtype="int64")
        # gt_indices[-1, :] -= 1  # there is no 'no_relation' category

        # foreground
        o1o2 = gt_classes[gt_indices[:2, :]].T
        for (o1, o2), gtr in zip(o1o2, gt_indices[2]):
            fg_matrix[o1 - 1, o2 - 1, gtr] += 1

    return fg_matrix
