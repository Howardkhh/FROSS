import pickle
from pathlib import Path
import json
from argparse import ArgumentParser
import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree
from tqdm import tqdm
import os
import shutil

def main(args):

    # Load data and mappings
    split = args.split
    scan_split = "validation" if split == "val" else split
    if args.label_categories == "scannet":
        scan_split_path = f"3DSSG_subset/{scan_split}_scans.txt"
        sg_ann_path = f"3DSSG_subset/relationships20.json"
        class_mapping_path = f"3DSSG_subset/3dssg_to_scannet.json"
        obj3d_ann_path = f"3DSSG/objects.json"
        OBJ_CLASS_NAME = "ScanNet_list"
        REL_CLASS_NAME = "ScanNet_rel"
        MAPPING_NAME = "3DSSG2NYUv2"
    elif args.label_categories == "replica":
        scan_split_path = f"ReplicaSSG/{scan_split}_scans.txt"
        sg_ann_path = f"ReplicaSSG/relationships.json"
        class_mapping_path = f"ReplicaSSG/replica_to_visual_genome.json"
        obj3d_ann_path = f"ReplicaSSG/objects.json"
        OBJ_CLASS_NAME = "VisualGenome_list"
        REL_CLASS_NAME = "VisualGenome_rel"
        MAPPING_NAME = "Replica2VisualGenome"

    with open(Path(args.dataset_path) / scan_split_path, 'r') as f:
        scan_ids = f.readlines()
    with open(Path(args.dataset_path) / obj3d_ann_path, 'r') as f:
        obj3d_ann = json.load(f) # Object annotations
    with open(Path(args.dataset_path) / sg_ann_path, 'r') as f:
        sg_ann = json.load(f) # Relationship annotations
    with open(Path(args.dataset_path) / class_mapping_path, 'r') as f:
        class_mapping = json.load(f) # Class mapping (3DSSG 140 classes to ScanNet 20 classes, NYU 27 classes to ScanNet 20 classes)
    with open(args.prediction_path, 'rb') as f:
        predictions = pickle.load(f) # Predictions from the model

    scan_ids = [scan_id.strip() for scan_id in scan_ids]
    scan_ids = scan_ids[:10] if args.debug else scan_ids
    scan2obj_rel = {} # Scan ID to object and relationship ground truth
    OBJID2IDX = {} # Scan ID to {Obejct ID to index in scan2obj_rel}
    obj_in_rel = {} # Scan ID to set of object IDs that are in relationships

    for scan in sg_ann["scans"]:
        if scan["scan"] not in scan_ids:
            continue
        obj_in_rel[scan["scan"]] = set()
        for rel in scan["relationships"]:
            s, o, _, _ = rel
            obj_in_rel[scan["scan"]].add(s)
            obj_in_rel[scan["scan"]].add(o)

    for scan in obj3d_ann["scans"]:
        if scan["scan"] not in scan_ids:
            continue
        scan2obj_rel[scan["scan"]] = {"obj_id": [], "obj_cls": [], "rel_edge": [], "rel_cls": []}
        OBJID2IDX[scan["scan"]] = {}
        for obj in scan["objects"]:
            if args.skip_no_rel_objects and int(obj["id"]) not in obj_in_rel[scan["scan"]]:
                continue
            if args.label_categories == "replica" and obj["label"] not in class_mapping[MAPPING_NAME]:
                continue # e.g. anonymize_text, unknown
            label = class_mapping[MAPPING_NAME][obj["label"]]
            if label not in class_mapping[OBJ_CLASS_NAME]:
                continue
            cls_idx = class_mapping[OBJ_CLASS_NAME].index(label)
            scan2obj_rel[scan["scan"]]["obj_cls"].append(cls_idx)
            scan2obj_rel[scan["scan"]]["obj_id"].append(int(obj["id"]))
            OBJID2IDX[scan["scan"]][int(obj["id"])] = len(scan2obj_rel[scan["scan"]]["obj_cls"]) - 1

    for scan in sg_ann["scans"]:
        if scan["scan"] not in scan_ids:
            continue
        for rel in scan["relationships"]:
            s, o, _, cls = rel
            cls_idx = class_mapping[REL_CLASS_NAME].index(cls)
            scan2obj_rel[scan["scan"]]["rel_edge"].append((OBJID2IDX[scan["scan"]][s], OBJID2IDX[scan["scan"]][o]))
            scan2obj_rel[scan["scan"]]["rel_cls"].append([cls_idx])

    ks = [1] # Recall@k
    topk = {"object": [], "relationship": [], "predicate": [],
            "object_per_class": {c: [] for c in range(len(class_mapping[OBJ_CLASS_NAME]))},
            "predicate_per_class": {c: [] for c in range(len(class_mapping[REL_CLASS_NAME]))}}
    
    total_pred_node = 0
    total_pred_edge = 0
    for scan_id in tqdm(scan_ids):
        node_gt = np.array(scan2obj_rel[scan_id]["obj_cls"]) # node_gt: clsIdx
        edge_gt = np.array(scan2obj_rel[scan_id]["rel_cls"]) # node_gt to node_gt: clsIdx
        edge_index_gt = np.array(scan2obj_rel[scan_id]["rel_edge"]) # node_gt to node_gt: edge_index

        gt_mesh_path = Path(args.dataset_path) / "data" / scan_id / f"labels.instances{'.align' if args.use_aligned_ply else ''}.annotated.v2.ply"
        gt_mesh = PlyData.read(str(gt_mesh_path))
        gt_points = np.stack([gt_mesh["vertex"]["x"], gt_mesh["vertex"]["y"], gt_mesh["vertex"]["z"]], axis=1)
        gt_point_ids = gt_mesh["vertex"]["objectId"]
        gt_points = gt_points[np.isin(gt_point_ids, list(OBJID2IDX[scan_id].keys()))]
        gt_point_ids = gt_point_ids[np.isin(gt_point_ids, list(OBJID2IDX[scan_id].keys()))]

        pred = predictions[scan_id]
        gt2pred = -np.ones((2, len(node_gt)), dtype=int) # node_gt to node: edge_index
        gt2pred[0] = np.arange(len(node_gt))
        if len(pred["cls"]) > 0:
            # Match ground truth objects to predicted objects (Section 4.1.3)
            overlap_count = np.zeros((len(node_gt), len(pred["cls"])))
            gt_kdtree = KDTree(gt_points)
            for pred_idx, seg in enumerate(pred["pcd"]):
                pred_num_points = len(seg)
                distances, indices = gt_kdtree.query(seg, distance_upper_bound=args.eval_overlap_threshold)
                matched_gt_idx = np.array([OBJID2IDX[scan_id][gt_point_ids[i]] for i in indices[indices != gt_kdtree.n]])
                for gt_idx in range(len(node_gt)):
                    overlap_count[gt_idx, pred_idx] = np.count_nonzero(matched_gt_idx == gt_idx)
                overlap_percentage = overlap_count[:, pred_idx] / pred_num_points
                sorted_gt_idx = np.flip(np.argsort(overlap_count[:, pred_idx], kind='stable'))
                max_gt_idx = sorted_gt_idx[0]
                second_gt_idx = sorted_gt_idx[1]
                if overlap_percentage[max_gt_idx] < 0.5 or overlap_percentage[second_gt_idx] / overlap_percentage[max_gt_idx] > 0.75:
                    overlap_count[:, pred_idx] = 0
                else:
                    overlap_count[np.arange(len(node_gt)) != max_gt_idx, pred_idx] = 0
            
            for gt_idx in range(len(node_gt)):
                max_pred_idx = np.argmax(overlap_count[gt_idx])
                if overlap_count[gt_idx, max_pred_idx] == 0:
                    continue
                gt2pred[1, gt_idx] = max_pred_idx # GT object to predicted object

        node_pred = pred["cls"] # node: pd

        edge_pred = pred["edge_cls"] # node to node: pd
        edge_index_pred = pred["edge_index"] # node to node: edge_index
        edge_index_pred_list = edge_index_pred.transpose().tolist()
        
        gt2pred_map = {}
        if gt2pred.shape[0] > 0:
            for i in range(gt2pred.shape[1]):
                gt_idx = gt2pred[0, i].item()
                pred_idx = gt2pred[1, i].item()
                gt2pred_map[gt_idx] = pred_idx

        # Below recall calculation codes are based on Wu et al. 2023 (https://github.com/ShunChengWu/3DSSG/blob/4b783ec/ssg/utils/util_eva.py).
        # Object Recall
        total_pred_node += len(node_pred)
        for i in range(len(node_gt)):
            gt_idx = gt2pred[0, i]
            pred_idx = gt2pred[1, i]
            if pred_idx < 0:
                topk["object"].append(99999)
                topk["object_per_class"][node_gt[gt_idx]].append(99999)
                continue
            pred = node_pred[pred_idx]
            gt = node_gt[gt_idx]
            sorted_args = np.flip(np.argsort(pred, kind='stable'))
            index = np.nonzero(sorted_args == gt)[0].item()
            topk["object"].append(index)
            topk["object_per_class"][gt].append(index)

        # Predicate Recall
        total_pred_edge += len(edge_index_pred_list)
        for i in range(len(edge_gt)):
            gt_rel = edge_gt[i]
            if len(gt_rel) != 1: print(len(gt_rel))
            gt_rel = gt_rel[0]
            sub_idx_gt = edge_index_gt[i, 0].item()
            obj_idx_gt = edge_index_gt[i, 1].item()
            sub_idx_pred = gt2pred_map[sub_idx_gt]
            obj_idx_pred = gt2pred_map[obj_idx_gt]

            if [sub_idx_pred, obj_idx_pred] not in edge_index_pred_list:
                topk["predicate"].append(99999)
                topk["predicate_per_class"][gt_rel].append(99999)
                continue
            pred_idx = edge_index_pred_list.index([sub_idx_pred, obj_idx_pred])
            pred_rel = edge_pred[pred_idx]

            sorted_args = np.flip(np.argsort(pred_rel, kind='stable'))
            index = np.nonzero(sorted_args == gt_rel)[0].item()
            topk["predicate"].append(index)
            topk["predicate_per_class"][gt_rel].append(index)

        # Relationship Recall
        for i in range(len(edge_gt)):
            gt_rel = edge_gt[i]
            if len(gt_rel) != 1: print(len(gt_rel))
            gt_rel = gt_rel[0]
            sub_idx_gt = edge_index_gt[i, 0].item()
            obj_idx_gt = edge_index_gt[i, 1].item()
            sub_idx_pred = gt2pred_map[sub_idx_gt]
            obj_idx_pred = gt2pred_map[obj_idx_gt]

            if [sub_idx_pred, obj_idx_pred] not in edge_index_pred_list:
                topk["relationship"].append(99999)
                continue
            
            pred_idx = edge_index_pred_list.index([sub_idx_pred, obj_idx_pred])
            pred_rel = edge_pred[pred_idx]
            pred_sub = node_pred[sub_idx_pred]
            pred_obj = node_pred[obj_idx_pred]

            gt_sub = node_gt[sub_idx_gt]
            gt_obj = node_gt[obj_idx_gt]

            so_preds = np.einsum('n,m->nm', pred_sub, pred_obj)
            conf_matrix = np.einsum('nm, k->nmk', so_preds, pred_rel) # confidences of all possible relationships
            _, cls_n, rel_k = conf_matrix.shape
            conf_matrix = conf_matrix.flatten()
            sorted_args = np.flip(np.argsort(conf_matrix, kind='stable'))

            gt_index = (gt_sub * cls_n + gt_obj) * rel_k + gt_rel
            index = np.nonzero(sorted_args == gt_index)[0].item()
            topk["relationship"].append(index)

    if args.output_path is None:
        try:
            terminal_width = shutil.get_terminal_size().columns
        except:
            terminal_width = 100

        print(f"Evaluation threshold: {args.eval_overlap_threshold}")
        print("Object:")
        for k in ks:
            print(f"\tRecall@{k}: {sum([1 for i in topk['object'] if i < k]) / len(topk['object'])}")
            print(f"\tPrecision@{k}: {sum([1 for i in topk['object'] if i < k]) / total_pred_node}")
            

        print("Predicate:")
        for k in ks:
            print(f"\tRecall@{k}: {sum([1 for i in topk['predicate'] if i < k]) / len(topk['predicate'])}")
            print(f"\tPrecision@{k}: {sum([1 for i in topk['predicate'] if i < k]) / total_pred_edge}")
            
        print("Relationship:")
        for k in ks:
            print(f"\tRecall@{k}: {sum([1 for i in topk['relationship'] if i < k]) / len(topk['relationship'])}")
            print(f"\tPrecision@{k}: {sum([1 for i in topk['relationship'] if i < k]) / total_pred_edge}")


        print("------------------------")

        if args.label_categories == "scannet":
            class_short = ["bathtub", "bed", "bookshelf", "cabinet", "chair", "counter", "curtain", "desk", "door", "floor", "otherfurniture", "picture", "refridgerator", "shower curtain", "sink", "sofa", "table", "toilet", "wall", "window"]
        elif args.label_categories == "replica":
            class_short = ["bag", "bskt.", "bed", "bench", "bike", "book", "botl.", "bowl", "box", "cab.", "chair", "clock", "cntr.", "cup", "curt.", "desk", "door", "lamp", "pil.", "plant", "plate", "pot", "rail.", "scrn.", "shlf.", "shoe", "sink", "stand", "table", "toil.", "towel", "umb.", "vase", "wind."]
        col_width = max(len(name) for name in class_short) + 2
        num_cols = max(1, terminal_width // col_width)

        print("Object per class (recall@1):")

        for c in range(len(class_short)):
            print(f"{class_short[c]:<{col_width}}", end="")
            if (c + 1) % num_cols == 0:
                print()
        print("avg.")

        i = 0
        for c in range(len(class_mapping[OBJ_CLASS_NAME])):
            if len(topk['object_per_class'][c]) == 0:
                continue
            print(f"{sum([1 for i in topk['object_per_class'][c] if i < 1]) / len(topk['object_per_class'][c]):<{col_width}.3f}", end="")
            if (i + 1) % num_cols == 0:
                print()
            i += 1
        print(f"{sum([sum([1 for i in topk['object_per_class'][c] if i < 1]) / len(topk['object_per_class'][c]) for c in range(len(class_mapping[OBJ_CLASS_NAME])) if len(topk['object_per_class'][c]) > 0]) / sum([1 for c in topk['object_per_class'] if len(topk['object_per_class'][c]) > 0]):<{col_width}.3f}")

        print("------------------------")

        if args.label_categories == "scannet":
            predicate_short = ["attached to", "build in", "connected to", "hanging on", "part of", "standing on", "supported by"]
        elif args.label_categories == "replica":
            predicate_short = ["above", "against", "attached to", "has", "in", "near", "on", "under", "with"]
        col_width = max(len(name) for name in predicate_short) + 2
        num_cols = max(1, terminal_width // col_width)

        print("Predicate per class (recall@1):")

        i = 0
        for c in range(len(class_mapping[REL_CLASS_NAME])):
            if len(topk['predicate_per_class'][c]) == 0:
                continue
            print(f"{class_mapping[REL_CLASS_NAME][c]:<{col_width}}", end="")
            if (i + 1) % num_cols == 0:
                print()
            i += 1
        print("avg.")

        i = 0
        for c in range(len(class_mapping[REL_CLASS_NAME])):
            if len(topk['predicate_per_class'][c]) == 0:
                continue
            print(f"{sum([1 for i in topk['predicate_per_class'][c] if i < 1]) / len(topk['predicate_per_class'][c]):<{col_width}.3f}", end="")
            if (i + 1) % num_cols == 0:
                print()
            i += 1
        print(f"{sum([sum([1 for i in topk['predicate_per_class'][c] if i < 1]) / len(topk['predicate_per_class'][c]) for c in range(len(class_mapping[REL_CLASS_NAME])) if len(topk['predicate_per_class'][c]) > 0]) / sum([1 for c in topk['predicate_per_class'] if len(topk['predicate_per_class'][c]) > 0]):<{col_width}.3f}")

        print("------------------------")

    else:
        os.makedirs(args.output_path.parent, exist_ok=True)
        with open(args.output_path, 'a') as f:
            f.write("\n")
            f.write(f"Evaluation threshold: {args.eval_overlap_threshold}\n")
            f.write("Object:\n")
            for k in ks:
                f.write(f"\tRecall@{k}: {sum([1 for i in topk['object'] if i < k]) / len(topk['object'])}\n")

            f.write("Predicate:\n")
            for k in ks:
                f.write(f"\tRecall@{k}: {sum([1 for i in topk['predicate'] if i < k]) / len(topk['predicate'])}\n")

            f.write("Relationship:\n")
            for k in ks:
                f.write(f"\tRecall@{k}: {sum([1 for i in topk['relationship'] if i < k]) / len(topk['relationship'])}\n")

            f.write("------------------------\n")

            if args.label_categories == "scannet":
                class_short = ["bathtub", "bed", "bookshelf", "cabinet", "chair", "counter", "curtain", "desk", "door", "floor", "otherfurniture", "picture", "refridgerator", "shower curtain", "sink", "sofa", "table", "toilet", "wall", "window"]
            elif args.label_categories == "replica":
                class_short = ["bag", "bskt.", "bed", "bench", "bike", "book", "botl.", "bowl", "box", "cab.", "chair", "clock", "cntr.", "cup", "curt.", "desk", "door", "lamp", "pil.", "plant", "plate", "pot", "rail.", "scrn.", "shlf.", "shoe", "sink", "stand", "table", "toil.", "towel", "umb.", "vase", "wind."]
            col_width = max(len(name) for name in class_short) + 2
            
            f.write("Object per class (recall@1):\n")

            for c in range(len(class_short)):
                f.write(f"{class_short[c]:<{col_width}}")
            f.write("avg.\n")

            for c in range(len(class_mapping[OBJ_CLASS_NAME])):
                if len(topk['object_per_class'][c]) == 0:
                    continue
                f.write(f"{sum([1 for i in topk['object_per_class'][c] if i < 1]) / len(topk['object_per_class'][c]):<{col_width}.3f}")
            f.write(f"{sum([sum([1 for i in topk['object_per_class'][c] if i < 1]) / len(topk['object_per_class'][c]) for c in range(len(class_mapping[OBJ_CLASS_NAME])) if len(topk['object_per_class'][c]) > 0]) / sum([1 for c in topk['object_per_class'] if len(topk['object_per_class'][c]) > 0]):<{col_width}.3f}\n")

            f.write("------------------------\n")

            if args.label_categories == "scannet":
                predicate_short = ["attached to", "build in", "connected to", "hanging on", "part of", "standing on", "supported by"]
            elif args.label_categories == "replica":
                predicate_short = ["above", "against", "attached to", "has", "in", "near", "on", "under", "with"]
            col_width = max(len(name) for name in predicate_short) + 2

            f.write("Predicate per class (recall@1):\n")

            for c in range(len(class_mapping[REL_CLASS_NAME])):
                if len(topk['predicate_per_class'][c]) == 0:
                    continue
                f.write(f"{class_mapping[REL_CLASS_NAME][c]:<{col_width}}")
            f.write("avg.\n")

            for c in range(len(class_mapping[REL_CLASS_NAME])):
                if len(topk['predicate_per_class'][c]) == 0:
                    continue
                f.write(f"{sum([1 for i in topk['predicate_per_class'][c] if i < 1]) / len(topk['predicate_per_class'][c]):<{col_width}.3f}")
            f.write(f"{sum([sum([1 for i in topk['predicate_per_class'][c] if i < 1]) / len(topk['predicate_per_class'][c]) for c in range(len(class_mapping[REL_CLASS_NAME])) if len(topk['predicate_per_class'][c]) > 0]) / sum([1 for c in topk['predicate_per_class'] if len(topk['predicate_per_class'][c]) > 0]):<{col_width}.3f}\n")

            f.write("------------------------\n")
            

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--label_categories", type=str, choices=["scannet", "replica"], default="scannet")
    args.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    args.add_argument("--prediction_path", type=Path, required=True)
    args.add_argument("--output_path", type=Path, default=None)
    args.add_argument("--eval_overlap_threshold", type=float, default=0.1)
    args.add_argument("--skip_no_rel_objects", action="store_true")
    args.add_argument("--use_aligned_ply", action="store_true") # for evaluating 3DSSG (Wu et. al, 2023) outputs
    args.add_argument("--debug", action="store_true", default=False)
    args = args.parse_args()

    assert not (args.label_categories == "replica" and args.use_aligned_ply), "Aligned ply is not available for Replica"

    main(args)