import argparse
import pickle
import os
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm
from PIL import Image
from classes import *
import matplotlib.pyplot as plt
from collections import defaultdict
import math

def draw_bbox(axe, bbox, label, score):
    score = score[score > 0.7]
    for i in range(len(bbox)):
        cx, cy, w, h = bbox[i]
        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        l = label[i]
        if l not in valid_class:
            continue
        color = obj_colors[valid_class.index(l)]
        L = 0.2126*(color[0]**2.2) + 0.7152*(color[1]**2.2) + 0.0722*(color[2]**2.2)
        axe.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=0.75))
        axe.text(x1+8, y1+8, 
                 f"{CLASSES[label[i]]}: {score[i]:.2f}", color="black" if L > 0.55 else "white", 
                 fontsize=3, fontfamily="sans-serif", ha='left', va='top', 
                 bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.2",
                           facecolor=color, edgecolor=color, alpha=1))

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

def draw_rel(axe, rel, rel_label, bbox, label):
    base_curvature=0.12
    grouped = defaultdict(list)
    for r, l in zip(rel, rel_label):
        grouped[(r[0], r[1])].append((l))

    for (si, oi), rels in grouped.items():
        # compute per-pair curvature values
        curvs = curvature_seq(len(rels), base_curvature)

        for idx, rel in enumerate(rels):
            sx = bbox[si][0]
            sy = bbox[si][1]
            ox = bbox[oi][0]
            oy = bbox[oi][1]

            curvature = curvs[idx]
            if rel not in valid_rel_class or si == oi or label[si] not in valid_class or label[oi] not in valid_class:
                continue
            rel_color = rel_colors[valid_rel_class.index(rel)]

            # Draw curved arrow
            arrow = FancyArrowPatch(
                (sx, sy), (ox, oy),
                connectionstyle=f"arc3,rad={curvature}",
                arrowstyle='-|>',
                mutation_scale=6,
                linewidth=0.5,
                color=rel_color,
            )
            axe.add_patch(arrow)
            lx, ly = arc3_midpoint((sx, sy), (ox, oy), curvature)

            L = 0.2126*(rel_color[0]**2.2) + 0.7152*(rel_color[1]**2.2) + 0.0722*(rel_color[2]**2.2)
            axe.text(
                lx, ly,
                REL_CLASSES[rel],
                ha="center", va="center",
                fontsize=2.5, color="black" if L > 0.55 else "white", fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.2",
                          facecolor=rel_color, edgecolor=rel_color)
            )
def main(args):
    scene = args.scene
    dataset_path = f"{args.dataset_path}/{scene}/sequence/"
    dataset = "ReplicaSSG" if "Replica" in dataset_path else "3RScan"
    print(f"Visualizing 2D SG for scene: {scene}")

    with open(f"{args.vis_folder}/{dataset}/{scene}/obj_2d.pkl", "rb") as f:
        obj_2d = pickle.load(f)
    with open(f"{args.vis_folder}/{dataset}/{scene}/rel_2d.pkl", "rb") as f:
        rel_2d = pickle.load(f)

    imgs = sorted([img for img in os.listdir(dataset_path) if img.endswith('.color.jpg')])

    os.makedirs(f"{args.vis_folder}/2D/{scene}", exist_ok=True)

    for idx, img in enumerate(tqdm(imgs, desc=f"{scene}: ")):
        I = Image.open(os.path.join(dataset_path, img))

        bboxes = obj_2d[idx]
        rels = rel_2d[idx]
        obj_classes = bboxes['classes']
        obj_boxes = bboxes['bboxes']
        obj_scores = bboxes['scores']
        rel_classes = rels['rel_classes']
        rels = rels['rels']
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(I)
        draw_rel(ax, rels, rel_classes, obj_boxes, obj_classes)
        draw_bbox(ax, obj_boxes, obj_classes, obj_scores)
        ax.axis('off')
        fig.set_size_inches(I.size[0] / 300, I.size[1] / 300)  # 100 DPI is easy to scale
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{args.vis_folder}/2D/{scene}/{img[:-4]}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--scene", type=str, required=True)
    args.add_argument("--vis_folder", type=str, required=True, help="Folder to the saved visualization pkl files, and output folder for the rendered images.")
    args = args.parse_args()
    main(args)