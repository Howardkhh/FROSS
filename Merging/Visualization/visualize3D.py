import numpy as np
import pyvista as pv
from scipy.stats import multivariate_normal
from math import pi
import pickle
from tqdm import tqdm
import os
import argparse
from classes import *

def main(args):
    scene = args.scene
    dataset_path = f"{args.dataset_path}/{scene}/sequence"
    print(f"Visualizing 3D Gaussians for scene: {scene}")

    with open(f"{args.vis_folder}/{scene}_obj.pkl", "rb") as f:
        obj = pickle.load(f)
    with open(f"{args.vis_folder}/{scene}_rel.pkl", "rb") as f:
        rel = pickle.load(f)

    scene_mesh = pv.read(f"{args.dataset_path}/{scene}/mesh.ply")

    imgs = sorted([img for img in os.listdir(dataset_path) if img.endswith('.color.jpg')])
    poses = sorted([pose for pose in os.listdir(dataset_path) if pose.endswith('.pose.txt') and not pose.endswith('slam.pose.txt')])
    assert len(imgs) == len(poses), f"Number of images and poses do not match: {len(imgs)} vs {len(poses)}"
    camera_rot = np.ndarray((len(poses), 3, 3))
    camera_trans = np.ndarray((len(poses), 3))
    for idx, pose_file in enumerate(poses):
        with open(f"{dataset_path}/{pose_file}") as f:
            extrinsic = np.array([list(map(float, line.strip().split())) for line in f])
        camera_rot[idx] = extrinsic[:3, :3] 
        camera_trans[idx] = extrinsic[:3, 3]

    os.makedirs(f"{args.vis_folder}/3D/{scene}", exist_ok=True)

    assert len(imgs) == len(obj) == len(rel) == len(camera_rot) == len(camera_trans), f"Length mismatch: {len(imgs)}, {len(obj)}, {len(rel)}, {len(camera_rot)}, {len(camera_trans)}"

    for idx in tqdm(range(len(imgs)), desc=f"{scene}: "):
        img = imgs[idx]
        pl = pv.Plotter(off_screen=True, window_size=(960, 540))
        pl.camera.view_angle = 58.4
        pl.camera.focal_point = (camera_trans[idx] + camera_rot[idx, :, 2]).tolist()
        pl.camera.position = camera_trans[idx].tolist()
        pl.camera.up = (-camera_rot[idx, :, 1]).tolist()
        pl.add_mesh(scene_mesh, rgb=True)

        classes = obj[idx]["classes"]
        means = obj[idx]["means"]
        covs = obj[idx]["covs"]

        for i in range(len(classes)):
            if classes[i] not in valid_class:
                continue
            color = obj_colors[valid_class.index(classes[i])]
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

            x, y, z = np.mgrid[mean[0]-radii_scaled:mean[0]+radii_scaled:step, mean[1]-radii_scaled:mean[1]+radii_scaled:step, mean[2]-radii_scaled:mean[2]+radii_scaled:step]
            pos = np.empty(x.shape + (3,))
            pos[:, :, :, 0] = x
            pos[:, :, :, 1] = y
            pos[:, :, :, 2] = z
            rv = multivariate_normal(mean, cov)
            pdf_values = rv.pdf(pos)
            pdf_values = pdf_values * ((2 * pi)**3 * np.linalg.det(cov)) ** (1/2)

            grid = pv.StructuredGrid(x, y, z)
            grid.point_data['pdf'] = pdf_values.flatten(order="F")
            for iso in np.arange(0.5, 1.0, 0.01):
                iso_surface = grid.contour([iso])
                mesh = iso_surface.extract_geometry()
                if mesh.n_points == 0:
                    continue
                pl.add_mesh(mesh, opacity=iso**2*0.1, color=color, lighting=False)

        pl.render()
        pl.screenshot(f"{args.vis_folder}/3D/{scene}/{img[:-4]}.png")
        pl.close()



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--scene", type=str, required=True)
    args.add_argument("--vis_folder", type=str, required=True, help="Folder to the saved visualization pkl files, and output folder for the rendered images.")
    args = args.parse_args()
    main(args)