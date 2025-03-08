import os
import argparse
from pathlib import Path
import json
import pickle
import time

import numpy as np
from tqdm import tqdm
import torch
from pycocotools.coco import COCO

from sg_loader import SG_Loader, GT_SG_Loader
from keyframe_selection import PeriodicKeyframeSelector, SpatialKeyframeSelector, DynamicKeyframeSelector
from sg_prediction import SG_Predictor
from global_sg import GlobalSG_Gaussian


def main(args):

    print(f"Using dataset: {'3RScan' if args.label_categories == 'scannet' else 'ReplicaSSG'}, path: {args.dataset_path}")
    print(f"Using split: {args.split}")
    print("Hyperparameters:")
    print(f"\tObject threshold: {args.obj_thresh}")
    print(f"\tRelation topk: {args.rel_topk}")
    print(f"\tHellinger threshold: {args.hellinger_threshold}")
    if args.kf_strategy != "none":
        print(f"\tKeyframe strategy: {args.kf_strategy}")
    if args.kf_strategy == "periodic":
        print(f"\tKeyframe interval: {args.kf_interval}")
    elif args.kf_strategy == "spatial":
        print(f"\tKeyframe translation threshold: {args.kf_translation}")
        print(f"\tKeyframe rotation threshold: {args.kf_rotation}")
    elif args.kf_strategy == "dynamic":
        print(f"\tKeyframe translation threshold: {args.kf_translation}")
        print(f"\tKeyframe rotation threshold: {args.kf_rotation}")
        print(f"\tKeyframe IOU threshold: {args.kf_iou_thresh}")
    print(f"Using ground truth scene graphs: {args.use_gt_sg}")
    print(f"Using ground truth camera poses: {args.use_gt_pose}")
    print(f"Saving results to: {args.output_path}")

    split = args.split
    if args.label_categories == "scannet":
        obj_ann_path = f"2DSG20/{split}.json"
        rel_ann_path = f"2DSG20/rel.json"
        SSG_path = f"3DSSG_subset"
        class_mapping_path = f"{SSG_path}/3dssg_to_scannet.json"
        OBJ_CLASS_NAME = "ScanNet_list"
        REL_CLASS_NAME = "ScanNet_rel"
    elif args.label_categories == "replica":
        assert split == "test" or split == "val", "Replica is only for testing"
        obj_ann_path = f"2DSG/test.json"
        rel_ann_path = f"2DSG/rel.json"
        SSG_path = f"ReplicaSSG"
        class_mapping_path = f"{SSG_path}/replica_to_visual_genome.json"
        OBJ_CLASS_NAME = "VisualGenome_list"
        REL_CLASS_NAME = "VisualGenome_rel"

    with open(Path(args.dataset_path) / class_mapping_path) as f:
        class_mapping = json.load(f)
    obj_classes = class_mapping[OBJ_CLASS_NAME]
    rel_classes = class_mapping[REL_CLASS_NAME]
    colors = np.random.rand(len(obj_classes), 3)

    scan_split = "validation" if split == "val" else split
    with open(Path(args.dataset_path) / SSG_path / f"{scan_split}_scans.txt") as f:
        scan_ids = f.readlines()
    scan_ids = [scan_id.strip() for scan_id in scan_ids]

    if not args.use_gt_sg:
        sg_predictor = SG_Predictor(args)
    else:
        objCOCO = COCO(Path(args.dataset_path) / obj_ann_path)
        with open(Path(args.dataset_path) / rel_ann_path) as f:
            rel_ann = json.load(f)[split]
    
    predictions = {}

    merge_time = 0
    obj_time = 0
    rel_time = 0
    frame_time = 0
    frame_cnt = 0
    kf_cnt = 0

    for scan_id in tqdm(scan_ids):

        global_sg = GlobalSG_Gaussian(args.hellinger_threshold, len(obj_classes), len(rel_classes), colors)

        if args.use_gt_sg:
            sg_loader = GT_SG_Loader(scan_id, split, objCOCO, rel_ann, args)
        else:
            sg_loader = SG_Loader(scan_id, split, args)

        if args.kf_strategy == "none":
            keyframe_selector = PeriodicKeyframeSelector(1)
        elif args.kf_strategy == "periodic":
            keyframe_selector = PeriodicKeyframeSelector(args.kf_interval)
        elif args.kf_strategy == "spatial":
            keyframe_selector = SpatialKeyframeSelector(args.kf_translation, args.kf_rotation)
        elif args.kf_strategy == "dynamic":
            keyframe_selector = DynamicKeyframeSelector(args.kf_translation, args.kf_rotation, args.kf_iou_thresh, len(obj_classes))

        camera_intrinsics = sg_loader.color_intrinsic

        frame_start_time = time.time()
        for idx, data in enumerate(sg_loader):
            frame_cnt += 1
            if args.use_gt_sg:
                depth, classes, bboxes, relation_classes, rels, camera_rot, camera_trans = data
            else:
                img, depth, camera_rot, camera_trans = data
                start_time = time.time()
                obj_det_output, all_scores, classes, bboxes = sg_predictor.detect_objects(img)
                obj_time += time.time() - start_time

            if not keyframe_selector.is_keyframe(camera_trans, camera_rot, classes):
                continue

            if not args.use_gt_sg:
                start_time = time.time()
                rels, relation_classes = sg_predictor.extract_relations(obj_det_output, all_scores)
                rel_time += time.time() - start_time


            start_time = time.time()
            global_sg.update(classes, bboxes, rels, relation_classes, depth, camera_rot, camera_trans, camera_intrinsics)
            merge_time += time.time() - start_time
            kf_cnt += 1

        frame_time += time.time() - frame_start_time

        prediction = {"pcd": [], "cls": [], "edge_index": [], "edge_cls": []}

        new_points_xyz = np.ndarray((0, 3), dtype=np.float32)
        new_points_rgb = np.ndarray((0, 3), dtype=np.uint8)
        point_clouds = []
        for idx in range(global_sg.global_group.classes.shape[0]):
            assert not (global_sg.global_group.classes[idx] < 0 or \
                        np.isnan(global_sg.global_group.means[idx]).any() or \
                        np.isnan(global_sg.global_group.covs[idx]).any() or \
                        (global_sg.global_group.rels[idx] < 0).any() or \
                        global_sg.global_group.pcd[idx] is None), \
                        f"class: {global_sg.global_group.classes[idx]<0}, mean: {np.isnan(global_sg.global_group.means[idx]).any()}, cov: {np.isnan(global_sg.global_group.covs[idx]).any()}, rels: {(global_sg.global_group.rels[idx] < 0).any()}, pcd: {global_sg.global_group.pcd[idx] is None}"
            pred_points = global_sg.global_group.pcd[idx]
            point_clouds.append(pred_points)
            color = np.random.randint(0, 255, 3)
            new_points_xyz = np.concatenate((new_points_xyz, np.array(pred_points)), axis=0)
            new_points_rgb = np.concatenate((new_points_rgb, np.full((len(pred_points), 3), color)), axis=0)
        prediction["pcd"] = point_clouds      
        prediction["cls"] = torch.nn.functional.one_hot(torch.tensor(global_sg.global_group.classes), len(obj_classes)).cpu().numpy()
        prediction["mean"] = global_sg.global_group.means
        prediction["cov"] = global_sg.global_group.covs
        s, o = np.nonzero(np.sum(global_sg.global_group.rels, axis=-1))
        prediction["edge_index"] = np.array([s, o])
        prediction["edge_cls"] = np.array(global_sg.global_group.rels[s, o] / np.sum(global_sg.global_group.rels[s, o], axis=-1, keepdims=True))
    
        predictions[scan_id] = prediction


    os.makedirs(args.output_path / args.label_categories, exist_ok=True)
    obj_name = f"obj{args.obj_thresh}"
    rel_name = f"rel{args.rel_topk}"
    hell_name = f"hell{args.hellinger_threshold}"
    if args.kf_strategy == "none":
        kf_name = "kfnone"
    elif args.kf_strategy == "periodic":
        kf_name = f"kfperiodic{args.kf_interval}"
    elif args.kf_strategy == "spatial":
        kf_name = f"kfspatial{args.kf_translation}_{args.kf_rotation}"
    elif args.kf_strategy == "dynamic":
        kf_name = f"kfdynamic{args.kf_translation}_{args.kf_rotation}_{args.kf_iou_thresh}"

    output_filename = f"predictions_gaussian_{obj_name}_{rel_name}_{hell_name}_{kf_name}_{args.split}{'_gt2dsg' if args.use_gt_sg else ''}{'_gtpose' if args.use_gt_pose else ''}.pkl"
    output_path = args.output_path / args.label_categories / output_filename
    with open(output_path, "wb") as f:
        pickle.dump(predictions, f)
    
    merge_time_per_frame = merge_time / kf_cnt
    obj_time_per_frame = obj_time / frame_cnt
    rel_time_per_frame = rel_time / kf_cnt
    print(f"frames: {frame_cnt}, keyframes: {kf_cnt}")
    print(f"Merge time per frame: {merge_time_per_frame}")
    print(f"Object time per frame: {obj_time_per_frame}")
    print(f"Relation time per frame: {rel_time_per_frame}")
    print(f"FPS: {frame_cnt / frame_time}")
    print(f"KF ratio: {kf_cnt / frame_cnt}")

    os.makedirs(args.output_path / "results" / args.label_categories, exist_ok=True)
    output_path = args.output_path / "results" / args.label_categories / output_filename
    with open(output_path, "w") as f:
        f.write(f"frames: {frame_cnt}, keyframes: {kf_cnt}\n")
        f.write(f"Merge time per frame: {merge_time_per_frame}\n")
        f.write(f"Object time per frame: {obj_time_per_frame}\n")
        f.write(f"Relation time per frame: {rel_time_per_frame}\n")
        f.write(f"FPS: {frame_cnt / frame_time}\n")
        f.write(f"KF ratio: {kf_cnt / frame_cnt}\n")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--artifact_path", type=Path)
    args.add_argument("--label_categories", type=str, choices=["scannet", "replica"], default="scannet")
    args.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    args.add_argument("--use_gt_sg", action="store_true", default=False)
    args.add_argument("--not_use_gt_pose", action="store_true", default=False)
    args.add_argument("--output_path", type=Path, default="output/")
    args.add_argument("--obj_thresh", type=float, default=0.7)
    args.add_argument("--rel_topk", type=int, default=10)
    args.add_argument("--hellinger_threshold", type=float, default=0.85)
    args.add_argument("--kf_strategy", type=str, default="none", choices=["none", "periodic", "spatial", "dynamic"])
    args.add_argument("--kf_interval", type=int, default=1, help="Periodic keyframe interval")
    args.add_argument("--kf_translation", type=float, default=0.01, help="Spatial keyframe translation threshold in meters")
    args.add_argument("--kf_rotation", type=float, default=0.017, help="Spatial keyframe rotation threshold in radians")
    args.add_argument("--kf_iou_thresh", type=float, default=0.2, help="Dynamic keyframe IOU threshold")
    args = args.parse_args()
    args.use_gt_pose = not args.not_use_gt_pose

    assert args.artifact_path is not None or args.use_gt_sg, "Artifact path is required when not using ground truth scene graphs"

    main(args)