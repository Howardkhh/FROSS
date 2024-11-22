import pickle
from pathlib import Path
import json
from argparse import ArgumentParser
import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree
from tqdm import tqdm

np.set_printoptions(linewidth=200, suppress=True)


def main(args):
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
        obj3d_ann = json.load(f)
    with open(Path(args.dataset_path) / sg_ann_path, 'r') as f:
        sg_ann = json.load(f)
    with open(Path(args.dataset_path) / class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    with open(args.prediction_path, 'rb') as f:
        predictions = pickle.load(f)

    scan_ids = [scan_id.strip() for scan_id in scan_ids]
    scan2obj_rel = {}
    OBJID2IDX = {}
    obj_in_rel = {}

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

    ks = [1]
    topk = {"object": [], "relationship": []}
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
            overlap_count = np.zeros((len(node_gt), len(pred["cls"])))
            gt_kdtree = KDTree(gt_points)
            for pred_idx, seg in enumerate(pred["pcd"]):
                pred_num_points = len(seg)
                distances, indices = gt_kdtree.query(seg, distance_upper_bound=args.eval_overlap_threshold)
                # print(distances.mean(), distances.std(), distances.min(), distances.max())
                # print(gt_point_ids[indices[indices != gt_kdtree.n]])
                matched_gt_idx = np.array([OBJID2IDX[scan_id][gt_point_ids[i]] for i in indices[indices != gt_kdtree.n]])
                # print(OBJID2IDX[scan_id])
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
                gt2pred[1, gt_idx] = max_pred_idx
            # print(scan_id)
            # print(overlap_count)
            # print([len(seg) for seg in pred["pcd"]])
            # print(gt2pred)
            # print(np.count_nonzero(gt2pred[1] >= 0) / len(node_gt))
            # if scan_id == "7272e17c-a01b-20f6-8b2f-e659331ae41a":
            #     print(gt_points.shape)
            #     points = []
            #     colors = np.random.randint(0, 255, (len(np.unique(gt_point_ids)), 3))
            #     for gt_idx, seg in enumerate(gt_points):
            #         points.append((*seg, *colors[OBJID2IDX[scan_id][gt_point_ids[gt_idx]]]))
            #     points_np = np.array(points, dtype=[
            #         ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
            #         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            #     ])
            #     PlyData([PlyElement.describe(points_np, 'vertex')]).write("7272e17c-a01b-20f6-8b2f-e659331ae41a_gt.ply")
            #     print([len(seg) for seg in pred["pcd"]])
            #     store pred segments to ply for visualization, each segment has its own color
            #     pred["pcd"] is a list of segments, each segment is a list of points (x, y, z)
                # points = []
                # import random
                # for pred_idx, seg in enumerate(pred["pcd"]):
                #     if pred_idx not in gt2pred[1]: continue
                #     r = random.randint(0, 255)
                #     g = random.randint(0, 255)
                #     b = random.randint(0, 255)
                #     for point in seg:
                #         points.append((*point, r, g, b))
                # points_np = np.array(points, dtype=[
                #     ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                #     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
                # ])
                # vis_ply_data = PlyData([PlyElement.describe(points_np, 'vertex')])
                # vis_ply_data.write(f"debug/{scan_id}.ply")
            #     exit()
                

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

        
        correct_count = 0
        for i in range(len(node_gt)):
            gt_idx = gt2pred[0, i]
            pred_idx = gt2pred[1, i]
            if pred_idx < 0:
                topk["object"].append(99999)
                continue
            pred = node_pred[pred_idx]
            gt = node_gt[gt_idx]
            sorted_args = np.flip(np.argsort(pred, kind='stable'))
            index = np.nonzero(sorted_args == gt)[0].item()
            topk["object"].append(index)
            if index == 0:
                correct_count += 1

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
            # print(f"gt: {class_mapping[OBJ_CLASS_NAME][gt_sub]} {class_mapping[REL_CLASS_NAME][gt_rel]} {class_mapping[OBJ_CLASS_NAME][gt_obj]}; pred: {class_mapping[OBJ_CLASS_NAME][np.argmax(pred_sub)]} {class_mapping[REL_CLASS_NAME][np.argmax(pred_rel)]} ({np.argmax(pred_rel)}) {class_mapping[OBJ_CLASS_NAME][np.argmax(pred_obj)]}")
            topk["relationship"].append(index)

    if args.output_path is None:
        print("Object:")
        for k in ks:
            print(f"Recall@{k}: {sum([1 for i in topk['object'] if i < k]) / len(topk['object'])}")

        print("Relationship:")
        for k in ks:
            print(f"Recall@{k}: {sum([1 for i in topk['relationship'] if i < k]) / len(topk['relationship'])}")
    else:
        with open(args.output_path, 'w') as f:
            f.write("Object: ")
            for k in ks:
                f.write(f"Recall@{k}: {sum([1 for i in topk['object'] if i < k]) / len(topk['object'])}\n")

            f.write("Relationship: ")
            for k in ks:
                f.write(f"Recall@{k}: {sum([1 for i in topk['relationship'] if i < k]) / len(topk['relationship'])}\n")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--label_categories", type=str, choices=["scannet", "replica"], default="scannet")
    args.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    args.add_argument("--prediction_path", type=Path, required=True)
    args.add_argument("--output_path", type=Path, default=None)
    args.add_argument("--eval_overlap_threshold", type=float, default=0.1)
    args.add_argument("--skip_no_rel_objects", action="store_true")
    args.add_argument("--use_aligned_ply", action="store_true")
    args = args.parse_args()

    assert not (args.label_categories == "replica" and args.use_aligned_ply), "Aligned ply is not available for Replica"

    main(args)