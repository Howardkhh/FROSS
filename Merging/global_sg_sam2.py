from math import floor
import time

import cv2
import numpy as np

import open3d as o3d

from utils import GaussianSG

# Global 3D scene graph with SAM2 masks
class GlobalSG_Gaussian_SAM2:
    def __init__(self, SAM2_predictor, hellinger_thershold, num_classes=20, num_rel_classes=7):
        self.SAM2_predictor = SAM2_predictor
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.global_group = GaussianSG(num_rel_classes, hellinger_thershold)

    def update(self, classes, bboxes, rels, rel_classes, depth, camera_rot, camera_trans, camera_intrinsic, img, downsample_ratio=1):
        """
        classes: (N,)
        bboxes:  (N,4) as [cx, cy, w, h] in pixel
        rels:    (E,2)
        rel_classes: (E,)
        depth:   HxW or HxWx1 (float) in meters
        camera_rot: (3,3)
        camera_trans: (3,) or (3,1)
        camera_intrinsic: obj with fx,fy,cx,cy
        img: BGR uint8 image (HxWx3)
        """

        sam2_time = 0
        project_time = 0
        compute_mean_cov_time = 0

        if len(classes) == 0:
            return sam2_time, project_time, compute_mean_cov_time

        camera_rot = camera_rot[None, ...] # (1, 3, 3)
        camera_trans = camera_trans[None, :, None] # (1, 3, 1)

        # filter out objects with invalid depth
        invalid_depth = depth[bboxes[:, 1], bboxes[:, 0]] == 0
        classes = classes[~invalid_depth]
        bboxes = bboxes[~invalid_depth]
        if len(rels) > 0:
            new_idx = np.cumsum(~invalid_depth) - 1
            new_idx[invalid_depth] = -1
            valid_edge_idx = np.logical_and(new_idx[rels[:, 0]] != -1, new_idx[rels[:, 1]] != -1)
            rel_classes = rel_classes[valid_edge_idx]
            rels = new_idx[rels[valid_edge_idx]]
        
        img = img.copy(order='C')
        # downsample image and depth for faster SAM2 processing
        if downsample_ratio != 1:
            img = cv2.resize(img, (img.shape[1]//downsample_ratio, img.shape[0]//downsample_ratio), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (depth.shape[1]//downsample_ratio, depth.shape[0]//downsample_ratio), interpolation=cv2.INTER_NEAREST)
            bboxes = bboxes / downsample_ratio

        self.SAM2_predictor.set_image(img)
        
        means, covs, pcds = [], [], []
        keep_mask = [] # record success masks

        # each bbox -> SAM2 mask -> 3D projection
        for i in range(len(bboxes)):
            start_time = time.time()

            cx, cy, w, h = bboxes[i] # center x,y, width, height
            x1 = int(np.round(cx - w/2)); y1 = int(np.round(cy - h/2))
            x2 = int(np.round(cx + w/2)); y2 = int(np.round(cy + h/2))

            # use bbox to get mask
            box = np.array([x1, y1, x2, y2], dtype=np.float32)[None, :]
            try:
                masks, scores, _ = self.SAM2_predictor.predict(box=box, multimask_output=True)
            except Exception as e:
                keep_mask.append(False)
                print(f"SAM2 prediction failed for object {i} with error: {e}")
                continue

            # If no valid masks, skip this object
            if masks is None or masks.shape[0] == 0:
                keep_mask.append(False)
                print(f"No valid SAM2 masks for object {i}.")
                continue
            # print(f"Object {i}: {masks.shape[0]} mask proposals from SAM2.")

            # Select the highest-scoring mask predicted by SAM2
            # Each box may have multiple mask proposals with confidence scores
            m = masks[np.argmax(scores)]  # Boolean mask of shape (H, W), where True = object region

            # Extract all pixel coordinates (y, x) where the mask is True
            ys, xs = np.where(m)  # ys: row indices (vertical), xs: column indices (horizontal)

            # If the mask has no valid pixels, skip this object
            if xs.size == 0:
                keep_mask.append(False)
                print(f"SAM2 mask for object {i} has no valid pixels.")
                continue

            sam2_time += time.time() - start_time
            start_time = time.time()

            # Stack the pixel coordinates into an (M, 2) array of [u, v] pairs,
            # where u = x (horizontal coordinate) and v = y (vertical coordinate)
            # These coordinates will be projected into 3D using the depth map
            points_2d = np.stack([xs, ys] * downsample_ratio, axis=1)  # (M, 2) pixel coordinates inside the object mask

            # project uv to 3D world coords

            # Eq. (2)
            x = (points_2d[:, 0] - camera_intrinsic.cx) / camera_intrinsic.fx
            y = (points_2d[:, 1] - camera_intrinsic.cy) / camera_intrinsic.fy
            z = np.ones_like(x)
            depth_val = depth[points_2d[:, 1], points_2d[:, 0], None] # (num_points, 1)
            camera_coord = (np.stack((x, y, z), axis=-1) * depth_val)[..., None] # (num_points, 3, 1)
            points_3d = (camera_rot @ camera_coord + camera_trans).squeeze(-1) # (num_points, 3)

            project_time += time.time() - start_time

            # ----------------------------------------------------------------------
            # Compute 3D mean and covariance of the projected points
            # ----------------------------------------------------------------------
            start_time = time.time()

            # mean_3d: object center in world coordinates
            # cov_3d: object 3D Gaussian covariance matrix
            if points_3d.shape[0] >= 3:
                # Mean: average of all 3D points along each dimension (x, y, z)
                mean_3d = np.mean(points_3d, axis=0)  # (3,)

                # Covariance: spatial spread of points in 3D
                # np.cov expects (variables along rows) → use rowvar=False
                cov_3d = np.cov(points_3d, rowvar=False)  # (3, 3)
            else:
                # If there are fewer than 3 valid points, use a fallback covariance
                mean_3d = np.mean(points_3d, axis=0)
                mins = np.min(points_3d, axis=0)
                maxs = np.max(points_3d, axis=0)
                lengths = np.maximum(maxs - mins, 1e-6)
                cov_3d = np.diag((lengths ** 2) / 12.0)  # approximate cuboid variance

            means.append(mean_3d)
            covs.append(cov_3d)
            pcds.append(points_3d)
            keep_mask.append(True)

            compute_mean_cov_time += time.time() - start_time

        keep_mask = np.array(keep_mask, dtype=bool)
        # if no successful masks, return
        if keep_mask.sum() == 0:
            return sam2_time, project_time, compute_mean_cov_time
        

        # Extract middle 50% x 50% of the bounding boxes for evaluation
        bboxes_xyxy_50 = np.concatenate((bboxes[:, :2] - bboxes[:, 2:] / 4, bboxes[:, :2] + bboxes[:, 2:] / 4), axis=1)
        proj_coords = []
        proj_count = []
        for x1, y1, x2, y2 in bboxes_xyxy_50:
            for x in range(int(x1), int(x2), 50):
                for y in range(int(y1), int(y2), 50):
                    proj_coords.append([x, y])
            proj_count.append((floor((int(x2)-int(x1))/50)+1) * (floor((int(y2)-int(y1))/50)+1))
        proj_coords = np.array(proj_coords)
        if len(proj_coords) == 0:
            proj_coords = np.zeros((0, 2), dtype=int)
        x = (proj_coords[:, 0] - camera_intrinsic.cx) / camera_intrinsic.fx
        y = (proj_coords[:, 1] - camera_intrinsic.cy) / camera_intrinsic.fy
        z = np.ones_like(x)
        depth_val = depth[proj_coords[:, 1], proj_coords[:, 0], None]
        camera_coord = (np.stack((x, y, z), axis=-1) * depth_val)[..., None]
        world_coord = (camera_rot @ camera_coord + camera_trans).squeeze(-1)
        pcds = np.split(world_coord, np.cumsum(proj_count)[:-1])

        # Add local 3D SG to global 3D SG
        update_idx = self.global_group.add(classes, mean_3d, cov_3d, rels, rel_classes, pcds)

        # Calculate the Hellinger distance and merge objects
        self.global_group.merge(update_idx)
        
        return sam2_time, project_time, compute_mean_cov_time