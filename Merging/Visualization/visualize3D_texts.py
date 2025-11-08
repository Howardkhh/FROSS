import numpy as np
import pickle
from tqdm import tqdm
import os
import argparse
from classes import *
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch
import math

@dataclass
class CameraIntrinsic():
    fx: float
    fy: float
    cx: float
    cy: float

def curvature_seq(k, base):
    seq = []
    m = 2
    sign = -1  # start bending upward; swap if you prefer
    while len(seq) < k:
        seq.append(sign * m * base)
        sign *= -1
        if sign == -1:
            m += 2
    return seq

def arc3_midpoint(p0, p1, rad):
    """
    p0, p1: 2D endpoints (sx, sy), (ox, oy)
    rad: curvature (same as arc3,rad=...)
    returns: (mx, my) midpoint on arc
    """
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    chord = p1 - p0
    d = np.linalg.norm(chord)
    if d == 0:
        return p0

    # straight midpoint
    mid = (p0 + p1) / 2

    if rad == 0:
        return mid

    # perpendicular direction (left-handed)
    perp = np.array([-chord[1], chord[0]]) / d

    # offset distance (same formula used internally by arc3)
    offset = rad * d / 2
    return mid + perp * offset

def main(args):
    scene = args.scene
    dataset_path = f"{args.dataset_path}/{scene}/sequence/"
    dataset = "ReplicaSSG" if "Replica" in dataset_path else "3RScan"
    image3D_path = f"{args.vis_folder}/3D/{scene}"
    print(f"Visualizing 3D SSG for scene: {scene}")
    
    with open(f"{args.vis_folder}/{dataset}/{scene}/obj.pkl", "rb") as f:
        obj = pickle.load(f)
    with open(f"{args.vis_folder}/{dataset}/{scene}/rel.pkl", "rb") as f:
        rel = pickle.load(f)

    imgs = sorted([img for img in os.listdir(image3D_path) if img.endswith('.color.png')])
    poses = sorted([pose for pose in os.listdir(dataset_path) if pose.endswith('.pose.txt') and not pose.endswith('slam.pose.txt')])

    camera_rot = np.ndarray((len(poses), 3, 3))
    camera_rot_inv = np.ndarray((len(poses), 3, 3))
    camera_trans = np.ndarray((len(poses), 3))
    for idx, pose_file in enumerate(poses):
        with open(f"{dataset_path}/{pose_file}") as f:
            extrinsic = np.array([list(map(float, line.strip().split())) for line in f])
        camera_rot[idx] = extrinsic[:3, :3] 
        camera_trans[idx] = extrinsic[:3, 3]
        camera_rot_inv[idx] = np.linalg.inv(camera_rot[idx])

    info_file = dataset_path + "_info.txt"
    with open(info_file) as f:
        infos = f.readlines()
    for info in infos:
        if info.startswith("m_colorWidth = "):
            color_w = int(info[len("m_colorWidth = "):])
        if info.startswith("m_colorHeight = "):
            color_h = int(info[len("m_colorHeight = "):])
        if info.startswith("m_depthWidth = "):
            depth_w = int(info[len("m_depthWidth = "):])
        if info.startswith("m_depthHeight = "):
            depth_h = int(info[len("m_depthHeight = "):])
        if info.startswith("m_depthShift = "):
            depth_shift = int(info[len("m_depthShift = "):])
        if info.startswith("m_calibrationColorIntrinsic = "):
            color_intrinsic = info[len("m_calibrationColorIntrinsic = "):].strip().split()
            color_intrinsic = CameraIntrinsic(fx=float(color_intrinsic[0]), fy=float(color_intrinsic[5]), cx=float(color_intrinsic[2]), cy=float(color_intrinsic[6]))
        if info.startswith("m_calibrationDepthIntrinsic = "):
            depth_intrinsic = info[len("m_calibrationDepthIntrinsic = "):].strip().split()
            depth_intrinsic = CameraIntrinsic(fx=float(depth_intrinsic[0]), fy=float(depth_intrinsic[5]), cx=float(depth_intrinsic[2]), cy=float(depth_intrinsic[6]))

    os.makedirs(f"{args.vis_folder}/3D_text/{scene}", exist_ok=True)

    for idx in tqdm(range(len(imgs)), desc=f"{scene}: "):
        img = imgs[idx]
        classes = obj[idx]["classes"]
        means = obj[idx]["means"]
        covs = obj[idx]["covs"]
        rels = rel[idx]

        I = Image.open(os.path.join(image3D_path, img))
        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.imshow(I)

        obj_mean = np.ones((len(classes), 2)) * -99999

        for i in range(len(classes)):
            if classes[i] not in valid_class:
                continue
            # Draw 3D bounding box
            mean = means[i]
            cov = covs[i]
            vals, vecs = np.linalg.eigh(cov)
            radii = np.sqrt(vals)            # 1σ radii
            scale = np.sqrt(2 * np.log(2))   # for half-maximum contour (~1.177)
            radii_scaled = (radii * scale).max()
            step = max(10, (2*radii_scaled/0.1)) * 1j
            if (mean[0]-radii_scaled < camera_trans[idx, 0] < mean[0]+radii_scaled) and \
               (mean[1]-radii_scaled < camera_trans[idx, 1] < mean[1]+radii_scaled) and \
               (mean[2]-radii_scaled < camera_trans[idx, 2] < mean[2]+radii_scaled):
                continue

            # project mean to 2d
            mean_cam = camera_rot_inv[idx] @ (mean - camera_trans[idx])
            if mean_cam[2] <= 0:
                continue
            mean_2d = mean_cam[:2] / mean_cam[2]
            mean_2d = np.array([mean_2d[0] * color_intrinsic.fx + color_intrinsic.cx, mean_2d[1] * color_intrinsic.fy + color_intrinsic.cy])
            if not (0 <= mean_2d[0] < color_w and 0 <= mean_2d[1] < color_h):
                continue

            obj_mean[i] = mean_2d
            



        topk = 10
        n = rels.shape[0]
        topk_cand = min(topk*5, n*n-1)
        rels_argmax = rels.argmax(-1)
        rels_max = rels[np.arange(n)[:, None], np.arange(n)[None, :], rels_argmax]
        rels_max_ravel = rels_max.ravel()
        top_idx = np.argpartition(-rels_max_ravel, topk_cand)[:topk_cand]
        top_idx = top_idx[rels_max_ravel[top_idx] > 0]
        top_idx = top_idx[np.argsort(-rels_max_ravel[top_idx])]
        top_i, top_j = np.unravel_index(top_idx, rels_max.shape)
        grouped = {}
        for i, j in zip(top_i, top_j):
            grouped[(i, j)] = int(rels_argmax[i, j])

        k = 0
        for (si, oi), r in grouped.items():
            if k >= topk or \
                r not in valid_rel_class or si == oi or \
                classes[si] not in valid_class or \
                classes[oi] not in valid_class or \
                (obj_mean[si] == -99999).any() or \
                (obj_mean[oi] == -99999).any():
                continue

            k += 1

            curvature = curvature_seq(1, 0.12)[0]
            rel_color = rel_colors[valid_rel_class.index(r)]
            sx, sy = obj_mean[si]
            ox, oy = obj_mean[oi]
            arrow = FancyArrowPatch(
                (sx, sy), (ox, oy),
                connectionstyle=f"arc3,rad={curvature}",
                arrowstyle='-|>',
                mutation_scale=6,
                linewidth=0.5,
                color=rel_color,
                shrinkA=7.5,
                shrinkB=7.5
            )
            axe.add_patch(arrow)

            lx, ly = arc3_midpoint((sx, sy), (ox, oy), curvature)
            
            if not (20 < lx < I.size[0]-20 and 20 < ly < I.size[1]-20): continue
            L = 0.2126*(rel_color[0]**2.2) + 0.7152*(rel_color[1]**2.2) + 0.0722*(rel_color[2]**2.2)
            axe.text(
                lx, ly,
                REL_CLASSES[r],
                ha="center", va="center",
                fontsize=2.5, color="black" if L > 0.55 else "white", fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.2",
                          facecolor=rel_color, edgecolor=rel_color)
            )

        for i in range(len(classes)):
            mean_2d = obj_mean[i]
            if (mean_2d == -99999).any(): continue
            if not (40 < mean_2d[0] < I.size[0]-40 and 20 < mean_2d[1] < I.size[1]-20): continue
            color = obj_colors[valid_class.index(classes[i])]
            # draw object tags
            L = 0.2126*(color[0]**2.2) + 0.7152*(color[1]**2.2) + 0.0722*(color[2]**2.2)
            axe.text(mean_2d[0], mean_2d[1], 
                    f"{CLASSES[classes[i]]}", color="black" if L > 0.55 else "white", 
                    fontsize=3, fontfamily="sans-serif", ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.2",
                            facecolor=color, edgecolor=color, alpha=1))

        axe.axis('off')
        fig.set_size_inches(I.size[0] / 300, I.size[1] / 300)  # 100 DPI is easy to scale
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{args.vis_folder}/3D_text/{scene}/{img[:-4]}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--scene", type=str, required=True)
    args.add_argument("--vis_folder", type=str, required=True, help="Folder to the saved visualization pkl files, and output folder for the rendered images.")
    args = args.parse_args()
    main(args)