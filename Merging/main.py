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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sg_loader import SG_Loader, GT_SG_Loader
from keyframe_selection import PeriodicKeyframeSelector, SpatialKeyframeSelector, DynamicKeyframeSelector
from sg_prediction import SG_Predictor
from global_sg import GlobalSG_Gaussian


def main(args):

    # Print configuration
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
    if args.use_kim:
        print(f"\tUsing Kim's merging method (kim et al., 2019)")
    if args.use_sam2:
        print(f"\tUsing SAM2")
        CKPT_PATH = "../sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        CFG_PATH  = "../sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = build_sam2(CFG_PATH, CKPT_PATH).to(device)
        SAM2_predictor = SAM2ImagePredictor(model)
        print("SAM2 model loaded.")


    print(f"Using ground truth scene graphs: {args.use_gt_sg}")
    print(f"Using ground truth camera poses: {args.use_gt_pose}")
    print(f"Saving results to: {args.output_path}")

    # Load dataset
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
        obj_ann_path = f"2DSG/{split}.json"
        rel_ann_path = f"2DSG/rel.json"
        SSG_path = f"ReplicaSSG"
        class_mapping_path = f"{SSG_path}/replica_to_visual_genome.json"
        OBJ_CLASS_NAME = "VisualGenome_list"
        REL_CLASS_NAME = "VisualGenome_rel"

    with open(Path(args.dataset_path) / class_mapping_path) as f:
        class_mapping = json.load(f)
    obj_classes = class_mapping[OBJ_CLASS_NAME]
    rel_classes = class_mapping[REL_CLASS_NAME]

    scan_split = "validation" if split == "val" else split
    with open(Path(args.dataset_path) / SSG_path / f"{scan_split}_scans.txt") as f:
        scan_ids = f.readlines()
    scan_ids = [scan_id.strip() for scan_id in scan_ids]
    scan_ids = scan_ids[:2] if args.debug else scan_ids

    if not args.use_gt_sg:
        sg_predictor = SG_Predictor(args)
    else:
        objCOCO = COCO(Path(args.dataset_path) / obj_ann_path)
        with open(Path(args.dataset_path) / rel_ann_path) as f:
            rel_ann = json.load(f)[split]

    # Inference Start
    predictions = {}

    merge_time = 0 # Time taken to merge local scene graphs into global scene graph
    sam2_total_time = 0 # Time taken for SAM2
    project_total_time = 0 # Time taken for projecting SAM2 masks to 3D
    compute_mean_cov_total_time = 0 # Time taken for computing mean and covariance of projected 3D points
    
    obj_time = 0 # Time taken for object detection
    rel_time = 0 # Time taken for relation extraction
    frame_time = 0 # Time taken for processing each frame
    frame_cnt = 0 # Total number of frames processed
    kf_cnt = 0 # Total number of keyframes selected

    for scan_id in tqdm(scan_ids):
        
        # Initialize global scene graph
        if not args.use_kim:
            if args.use_sam2:
                from global_sg_sam2 import GlobalSG_Gaussian_SAM2
                global_sg = GlobalSG_Gaussian_SAM2(SAM2_predictor, args.hellinger_threshold, len(obj_classes), len(rel_classes))
            else:
                global_sg = GlobalSG_Gaussian(args.hellinger_threshold, len(obj_classes), len(rel_classes))
        else:
            from global_sg_kim import GlobalSG_Kim
            global_sg = GlobalSG_Kim(args.hellinger_threshold, len(obj_classes), len(rel_classes))

        # Initialize scene graph loader
        if args.use_gt_sg:
            sg_loader = GT_SG_Loader(scan_id, split, objCOCO, rel_ann, args)
        else:
            sg_loader = SG_Loader(scan_id, split, args)

        # Initialize keyframe selector (experimental, not used in the paper)
        if args.kf_strategy == "none":
            keyframe_selector = PeriodicKeyframeSelector(1)
        elif args.kf_strategy == "periodic":
            keyframe_selector = PeriodicKeyframeSelector(args.kf_interval)
        elif args.kf_strategy == "spatial":
            keyframe_selector = SpatialKeyframeSelector(args.kf_translation, args.kf_rotation)
        elif args.kf_strategy == "dynamic":
            keyframe_selector = DynamicKeyframeSelector(args.kf_translation, args.kf_rotation, args.kf_iou_thresh, len(obj_classes))

        camera_intrinsics = sg_loader.color_intrinsic

        # Process each frame
        frame_start_time = time.time()
        for idx, data in enumerate(sg_loader):
            frame_cnt += 1
             
            # Detect objects
            if args.use_gt_sg:
                depth, classes, bboxes, relation_classes, rels, camera_rot, camera_trans = data
            else:
                img, depth, camera_rot, camera_trans = data
                start_time = time.time()
                obj_det_output, all_scores, classes, class_probs, bboxes = sg_predictor.detect_objects(img)
                obj_time += time.time() - start_time

            # Select keyframes (based on detected objects)
            if not keyframe_selector.is_keyframe(camera_trans, camera_rot, classes):
                continue
            
            # Extract relations
            if not args.use_gt_sg:
                start_time = time.time()
                rels, relation_classes = sg_predictor.extract_relations(obj_det_output, all_scores)
                rel_time += time.time() - start_time

            # Merge local scene graph into global scene graph
            start_time = time.time()
            input_classes = class_probs if args.use_kim else classes
            if args.use_kim:
                global_sg.update(input_classes, bboxes, rels, relation_classes, depth, camera_rot, camera_trans, camera_intrinsics, img)
            elif args.use_sam2:
                sam2_time, project_time, compute_mean_cov_time = global_sg.update(input_classes, bboxes, rels, relation_classes, depth, camera_rot, camera_trans, camera_intrinsics, img)
                sam2_total_time += sam2_time
                project_total_time += project_time
                compute_mean_cov_total_time += compute_mean_cov_time
            else:
                global_sg.update(input_classes, bboxes, rels, relation_classes, depth, camera_rot, camera_trans, camera_intrinsics)
            merge_time += time.time() - start_time

            kf_cnt += 1

        frame_time += time.time() - frame_start_time

        # Prepare predictions for the current scene
        prediction = {"pcd": [], "cls": [], "edge_index": [], "edge_cls": []}

        new_points_xyz = np.ndarray((0, 3), dtype=np.float32)
        new_points_rgb = np.ndarray((0, 3), dtype=np.uint8)
        point_clouds = []
        classes = global_sg.global_group.classes
        means = global_sg.global_group.means
        covs = global_sg.global_group.covs
        rels = global_sg.global_group.rels
        pcd = global_sg.global_group.pcd
        for idx in range(classes.shape[0]):
            # Sanity checks
            assert not ((classes[idx] < 0).any() or \
                        np.isnan(means[idx]).any() or \
                        np.isnan(covs[idx]).any() or \
                        (rels[idx] < 0).any() or \
                        pcd[idx] is None), \
                        f"class: {(classes[idx] < 0).any()}, mean: {np.isnan(means[idx]).any()}, cov: {np.isnan(covs[idx]).any()}, rels: {(rels[idx] < 0).any()}, pcd: {pcd[idx] is None}"
            
            pred_points = pcd[idx]
            if args.use_kim:
                pred_points = pred_points[::2500] # skip 50 * 50 to reduce file size
            point_clouds.append(pred_points)
            color = np.random.randint(0, 255, 3)
            new_points_xyz = np.concatenate((new_points_xyz, np.array(pred_points)), axis=0)
            new_points_rgb = np.concatenate((new_points_rgb, np.full((len(pred_points), 3), color)), axis=0)

        prediction["pcd"] = point_clouds
        if not args.use_kim: prediction["cls"] = torch.nn.functional.one_hot(torch.tensor(classes), len(obj_classes)).cpu().numpy()
        else: prediction["cls"] = classes
        prediction["mean"] = means
        prediction["cov"] = covs
        s, o = np.nonzero(np.sum(rels, axis=-1))
        prediction["edge_index"] = np.array([s, o])
        prediction["edge_cls"] = np.array(rels[s, o] / np.sum(rels[s, o], axis=-1, keepdims=True))
    
        predictions[scan_id] = prediction

    # Save predictions
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

    output_filename = f"predictions_gaussian_{obj_name}_{rel_name}_{hell_name}_{kf_name}_{args.split}{'_gt2dsg' if args.use_gt_sg else ''}{'_gtpose' if args.use_gt_pose else ''}{'_kim' if args.use_kim else ''}{'_sam2' if args.use_sam2 else ''}.pkl"
    output_path = args.output_path / args.label_categories / output_filename
    with open(output_path, "wb") as f:
        pickle.dump(predictions, f)

    merge_time_per_frame = merge_time / kf_cnt
    obj_time_per_frame = obj_time / frame_cnt # Object detection is done for each frame, not keyframe
    rel_time_per_frame = rel_time / kf_cnt
    print(f"Number of frames: {frame_cnt}, keyframes: {kf_cnt}")
    print(f"Merge time per frame: {merge_time_per_frame}")
    if args.use_sam2:
        print(f"SAM2 time per frame: {sam2_total_time / kf_cnt}")
        print(f"Projection time per frame: {project_total_time / kf_cnt}")
        print(f"Compute mean/cov time per frame: {compute_mean_cov_total_time / kf_cnt}")
    print(f"Object time per frame: {obj_time_per_frame}")
    print(f"Relation time per frame: {rel_time_per_frame}")
    print(f"FPS: {frame_cnt / frame_time}")
    print(f"KF ratio: {kf_cnt / frame_cnt}")

    os.makedirs(args.output_path / "results" / args.label_categories, exist_ok=True)
    output_path = args.output_path / "results" / args.label_categories / f"{output_filename[:-4]}.txt"
    with open(output_path, "w") as f:
        f.write(f"Number of frames: {frame_cnt}, keyframes: {kf_cnt}\n")
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
    args.add_argument("--kf_strategy", type=str, default="none", choices=["none", "periodic", "spatial", "dynamic"], 
                      help="Keyframe selection strategy. " \
                      "'none' means no keyframe selection, " \
                      "'periodic' selects every --kf_interval frames, " \
                      "'spatial' selects keyframes based on --kf_translation spatial translation and --kf_rotation rotation thresholds, " \
                      "'dynamic' selects keyframes dynamically based on detected object classes' IoU, spatial translation and rotation.")
    args.add_argument("--kf_interval", type=int, default=1, help="Periodic keyframe interval. Only used if --kf_strategy is 'periodic'.")
    args.add_argument("--kf_translation", type=float, default=0.01, help="Spatial keyframe translation threshold in meters. Only used if --kf_strategy is 'spatial' or 'dynamic'.")
    args.add_argument("--kf_rotation", type=float, default=0.017, help="Spatial keyframe rotation threshold in radians. Only used if --kf_strategy is 'spatial' or 'dynamic'.")
    args.add_argument("--kf_iou_thresh", type=float, default=0.2, help="Dynamic keyframe IoU threshold. Only used if --kf_strategy is 'dynamic'.")
    args.add_argument("--use_kim", action="store_true", default=False, help="Use Kim's merging method (Kim et al., 2019).")
    args.add_argument("--use_sam2", action="store_true", default=False, help="Use SAM2.")
    args.add_argument("--debug", action="store_true", default=False)

    args = args.parse_args()
    args.use_gt_pose = not args.not_use_gt_pose

    assert args.artifact_path is not None or args.use_gt_sg, "Artifact path is required when not using ground truth scene graphs"

    main(args)