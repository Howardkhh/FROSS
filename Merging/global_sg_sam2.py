from math import floor
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d  # kept for consistency, even if not used directly here

from utils import GaussianSG


class GlobalSG_Gaussian_SAM2:
    """
    Global 3D scene graph using Gaussian representations, augmented with SAM2 masks.

    This class:
      1. Takes 2D detections + relations in a single frame
      2. Uses SAM2 to refine each bbox into a segmentation mask
      3. Projects mask pixels into 3D using depth + camera intrinsics/extrinsics
      4. Computes a 3D Gaussian (mean + covariance) per object
      5. Adds / merges them into a global Gaussian scene graph
      6. Optionally saves per-object visualizations:
         [left: original image, right: object mask overlay in red with label]
    """

    def __init__(
        self,
        SAM2_predictor,
        hellinger_thershold,
        num_classes=20,
        num_rel_classes=7,
        visualize=False,
        sam2_mask_dir=None,
        class_names=None,
        remove_ambiguous_by_class=False,
    ):
        """
        Args:
            SAM2_predictor: SAM2 predictor object with .set_image() and .predict(...)
            hellinger_thershold: float, threshold used inside GaussianSG for merging
            num_classes: number of object classes
            num_rel_classes: number of relation classes
            visualize: if True, keep snapshots of global SG over time
            sam2_mask_dir: directory to save per-object visualizations. If None, no images are written.
            class_names: list of strings, mapping class_id -> human-readable name.
                         If None or out of range, we fall back to "class_{id}".
            remove_ambiguous_by_class: if True, any pixel that belongs to
                masks from 2 or more *different classes* will be removed from all masks.
        """
        self.SAM2_predictor = SAM2_predictor
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.global_group = GaussianSG(num_rel_classes, hellinger_thershold)

        self.visualize = visualize
        self.sam2_mask_dir = sam2_mask_dir
        if self.sam2_mask_dir is not None:
            self.sam2_mask_dir = Path(sam2_mask_dir)
            self.sam2_mask_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names  # optional list of class names
        self.remove_ambiguous_by_class = remove_ambiguous_by_class

        self.cur_obj = []
        self.cur_rel = []
        self._frame_idx = 0  # frame counter (used in filenames)

    def _class_id_to_name(self, class_id: int) -> str:
        """Map class_id to a readable string name."""
        if (
            self.class_names is not None
            and 0 <= class_id < len(self.class_names)
        ):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def update(
        self,
        classes,
        bboxes,
        rels,
        rel_classes,
        depth,
        camera_rot,
        camera_trans,
        camera_intrinsic,
        img,
    ):
        """
        Update the global 3D Gaussian scene graph with one RGB-D frame.

        Args:
            classes:       (N,) int array of object class indices
            bboxes:        (N, 4) float array [cx, cy, w, h] in pixel coordinates
            rels:          (E, 2) int array of (subject_idx, object_idx)
            rel_classes:   (E,) int array of relation class indices
            depth:         HxW or HxWx1 float depth map in meters
            camera_rot:    (3, 3) rotation matrix (world_from_camera)
            camera_trans:  (3,) or (3, 1) translation vector (world_from_camera)
            camera_intrinsic: object with attributes fx, fy, cx, cy
            img:           HxWx3 BGR uint8 image (OpenCV style)

        Returns:
            sam2_time:            total time spent in SAM2 calls
            project_time:         total time spent in 2D->3D projection
            compute_mean_cov_time: time spent computing 3D mean + covariance
        """
        sam2_time = 0.0
        project_time = 0.0
        compute_mean_cov_time = 0.0

        # If there are no detections, optionally record an empty snapshot and return
        if len(classes) == 0:
            if self.visualize:
                self.cur_obj.append(
                    {
                        "classes": self.global_group.classes.copy(),
                        "means": self.global_group.means.copy(),
                        "covs": self.global_group.covs.copy(),
                    }
                )
                self.cur_rel.append(self.global_group.rels.copy())
                self._frame_idx += 1
            return sam2_time, project_time, compute_mean_cov_time

        # Ensure rotation and translation have shapes (1, 3, 3) and (1, 3, 1)
        camera_rot = camera_rot[None, ...]  # (1, 3, 3)
        camera_trans = camera_trans.reshape(1, 3, 1)  # (1, 3, 1)

        # Filter out objects whose center has invalid depth (0)
        invalid_depth = depth[bboxes[:, 1], bboxes[:, 0]] == 0
        classes = classes[~invalid_depth]
        bboxes = bboxes[~invalid_depth]

        if len(rels) > 0:
            # Re-map relation indices after filtering invalid objects
            new_idx = np.cumsum(~invalid_depth) - 1
            new_idx[invalid_depth] = -1
            valid_edge_idx = np.logical_and(
                new_idx[rels[:, 0]] != -1, new_idx[rels[:, 1]] != -1
            )
            rel_classes = rel_classes[valid_edge_idx]
            rels = new_idx[rels[valid_edge_idx]]

        if len(classes) == 0:
            # No valid objects after removing invalid depth
            if self.visualize:
                self.cur_obj.append(
                    {
                        "classes": self.global_group.classes.copy(),
                        "means": self.global_group.means.copy(),
                        "covs": self.global_group.covs.copy(),
                    }
                )
                self.cur_rel.append(self.global_group.rels.copy())
                self._frame_idx += 1
            return sam2_time, project_time, compute_mean_cov_time


        # Make a copy of the image for visualization
        img = img.copy(order="C")
        orig_img = img.copy()  # "clean" image for side-by-side comparison

        # Set image for SAM2
        self.SAM2_predictor.set_image(img)

        obj_masks = []        # list of boolean masks, one per object
        invalid_objs = []      # list of booleans, True if object i is invalid
        # --------------------------------------------------------------
        # 1st pass: bbox -> SAM2 masks (no 3D projection yet)
        # --------------------------------------------------------------
        for i in range(len(bboxes)):
            start_time = time.time()

            cx_i, cy_i, w_i, h_i = bboxes[i]  # center x,y, width, height
            x1 = int(np.round(cx_i - w_i / 2))
            y1 = int(np.round(cy_i - h_i / 2))
            x2 = int(np.round(cx_i + w_i / 2))
            y2 = int(np.round(cy_i + h_i / 2))

            # Clip bbox to image boundaries
            x1 = max(0, min(x1, img.shape[1] - 1))
            x2 = max(0, min(x2, img.shape[1] - 1))
            y1 = max(0, min(y1, img.shape[0] - 1))
            y2 = max(0, min(y2, img.shape[0] - 1))

            # If box collapsed, skip
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid bbox for object {i}: [{x1}, {y1}, {x2}, {y2}]")
                invalid_objs.append(True)
                continue

            box = np.array([x1, y1, x2, y2], dtype=np.float32)[None, :]

            # Run SAM2 for this bbox
            try:
                masks, scores, _ = self.SAM2_predictor.predict(
                    box=box, multimask_output=True
                )
            except Exception as e:
                print(f"SAM2 prediction failed for object {i} with error: {e}")
                invalid_objs.append(True)
                continue

            # No valid mask
            if masks is None or masks.shape[0] == 0:
                print(f"No valid SAM2 masks for object {i}.")
                invalid_objs.append(True)
                continue

            # Choose highest-scoring mask
            best_mask = masks[np.argmax(scores)]  # (H, W) bool

            # If mask is empty, skip
            if not best_mask.any():
                print(f"SAM2 mask for object {i} has no valid pixels.")
                invalid_objs.append(True)
                continue

            sam2_time += time.time() - start_time

            obj_masks.append(best_mask)  
            invalid_objs.append(False)

        invalid_objs = np.array(invalid_objs, dtype=bool)
        classes = classes[~invalid_objs]
        bboxes = bboxes[~invalid_objs]

        if len(rels) > 0:
            # Re-map relation indices after filtering invalid objects
            new_idx = np.cumsum(~invalid_objs) - 1
            new_idx[invalid_objs] = -1
            valid_edge_idx = np.logical_and(
                new_idx[rels[:, 0]] != -1, new_idx[rels[:, 1]] != -1
            )
            rel_classes = rel_classes[valid_edge_idx]
            rels = new_idx[rels[valid_edge_idx]]
        

        obj_masks = np.stack(obj_masks, axis=0)  # (K, H, W) boolean
        K, H, W = obj_masks.shape

        # --------------------------------------------------------------
        # Optionally remove ambiguous pixels:
        #   Any pixel that is covered by >= 2 different classes is removed
        #   from ALL masks.
        # --------------------------------------------------------------
        if self.remove_ambiguous_by_class:
            # unique class IDs among remaining objects
            unique_cls = np.unique(classes)
            # class_cover[c] is True where any object of class 'unique_cls[c]' covers that pixel
            class_cover = []
            for cls_id in unique_cls:
                # masks belonging to this class
                cls_masks = obj_masks[classes == cls_id]  # shape (Nc, H, W)
                cls_union = np.any(cls_masks, axis=0)     # (H, W)
                class_cover.append(cls_union)
            class_cover = np.stack(class_cover, axis=0)   # (C, H, W)

            # For each pixel, count how many different classes cover it
            class_cover_count = class_cover.sum(axis=0)   # (H, W)
            ambiguous_pixels = class_cover_count > 1      # True where >1 class overlap

            if ambiguous_pixels.any():
                # Remove ambiguous pixels from all masks
                obj_masks[:, ambiguous_pixels] = False

            # ----------------------------------------------------------
            # Morphological opening + closing + optional erosion (shrink mask)
            # ----------------------------------------------------------
            kernel_size = 7     
            erode_size  = 5
            kernel_openclose = np.ones((kernel_size, kernel_size), np.uint8)
            kernel_erode     = np.ones((erode_size, erode_size), np.uint8)
            invalid_objs = []

            for k in range(obj_masks.shape[0]):
                # Convert boolean mask → uint8 (0/255)
                m = (obj_masks[k].astype(np.uint8) * 255)

                # Opening (remove noise)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_openclose)

                # Closing (fill holes)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_openclose)

                # ------------------------------------------------------
                #  ERODE: shrink mask inward by 1 "ring" (configurable)
                # ------------------------------------------------------
                m = cv2.erode(m, kernel_erode, iterations=1)

                obj_masks[k] = (m > 0)  # back to boolean   
                if not obj_masks[k].any():
                    invalid_objs.append(True)
                else:
                    invalid_objs.append(False)

            invalid_objs = np.array(invalid_objs, dtype=bool)
            obj_masks = obj_masks[~invalid_objs]
            classes = classes[~invalid_objs]
            bboxes = bboxes[~invalid_objs]

            if obj_masks.shape[0] == 0:
                # No objects left after removing ambiguous regions
                if self.visualize:
                    self.cur_obj.append(
                        {
                            "classes": self.global_group.classes.copy(),
                            "means": self.global_group.means.copy(),
                            "covs": self.global_group.covs.copy(),
                        }
                    )
                    self.cur_rel.append(self.global_group.rels.copy())
                    self._frame_idx += 1
                return sam2_time, project_time, compute_mean_cov_time

            K = obj_masks.shape[0]

            if len(rels) > 0:
                # Re-map relation indices after filtering invalid objects
                new_idx = np.cumsum(~invalid_objs) - 1
                new_idx[invalid_objs] = -1
                valid_edge_idx = np.logical_and(
                    new_idx[rels[:, 0]] != -1, new_idx[rels[:, 1]] != -1
                )
                rel_classes = rel_classes[valid_edge_idx]
                rels = new_idx[rels[valid_edge_idx]]

        # --------------------------------------------------------------
        # 2nd pass: for each mask -> visualize, project to 3D, compute mean/cov
        # --------------------------------------------------------------
        means = []
        covs = []
        pcds_for_eval = []

        for i in range(K):
            start_time = time.time()

            mask_i = obj_masks[i]  # (H, W) bool
            ys, xs = np.where(mask_i)
            if xs.size == 0:
                # Should not happen after the checks above, but keep safe
                continue

            # Visualization: overlay mask in red, draw bbox & label, save
            if self.sam2_mask_dir is not None:
                overlay = orig_img.copy()
                red = (0, 0, 255)  # BGR red
                alpha = 0.5

                overlay_region = overlay[ys, xs].astype(np.float32)
                red_arr = np.array(red, dtype=np.float32)
                blended = (1.0 - alpha) * overlay_region + alpha * red_arr
                overlay[ys, xs] = blended.astype(np.uint8)

                # Draw bbox
                cx_i, cy_i, w_i, h_i = bboxes[i]
                x1 = int(np.round(cx_i - w_i / 2))
                y1 = int(np.round(cy_i - h_i / 2))
                x2 = int(np.round(cx_i + w_i / 2))
                y2 = int(np.round(cy_i + h_i / 2))
                x1 = max(0, min(x1, overlay.shape[1] - 1))
                x2 = max(0, min(x2, overlay.shape[1] - 1))
                y1 = max(0, min(y1, overlay.shape[0] - 1))
                y2 = max(0, min(y2, overlay.shape[0] - 1))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), red, 2)

                # Put object label inside the box (top-left)
                class_id = int(classes[i])
                label = self._class_id_to_name(class_id)
                cv2.putText(
                    overlay,
                    label,
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                side_by_side = np.concatenate([orig_img, overlay], axis=1)
                out_path = self.sam2_mask_dir / f"frame_{self._frame_idx:06d}_obj_{i:03d}.png"
                cv2.imwrite(str(out_path), side_by_side)

            sam2_time += time.time() - start_time
            start_time = time.time()

            # 2D -> 3D projection
            u = xs
            v = ys

            x = (u - camera_intrinsic.cx) / camera_intrinsic.fx
            y = (v - camera_intrinsic.cy) / camera_intrinsic.fy
            z = np.ones_like(x)

            depth_val = depth[v, u, None]  # (N, 1)
            camera_coord = (np.stack((x, y, z), axis=-1) * depth_val)[..., None]  # (N, 3, 1)
            points_3d = (camera_rot @ camera_coord + camera_trans).squeeze(-1)  # (N, 3)

            project_time += time.time() - start_time
            start_time = time.time()

            # Compute 3D mean & covariance
            # if points_3d.shape[0] >= 3:
            if False:
                mean_3d = np.mean(points_3d, axis=0)  # (3,)
                cov_3d = np.cov(points_3d, rowvar=False)  # (3, 3)
            else:
                # Fallback: approximate covariance from extents
                mean_3d = np.mean(points_3d, axis=0)
                mins = np.min(points_3d, axis=0)
                maxs = np.max(points_3d, axis=0)
                lengths = np.maximum(maxs - mins, 1e-6)
                cov_3d = np.diag((lengths ** 2) / 12.0)

            means.append(mean_3d)
            covs.append(cov_3d)

            compute_mean_cov_time += time.time() - start_time

        means = np.stack(means, axis=0)  # (K', 3)
        covs = np.stack(covs, axis=0)    # (K', 3, 3)

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

        if self.visualize:
            self.cur_obj.append({"classes": self.global_group.classes.copy(), "means": self.global_group.means.copy(), "covs": self.global_group.covs.copy()})
            self.cur_rel.append(self.global_group.rels.copy())
        
        self._frame_idx += 1
        return sam2_time, project_time, compute_mean_cov_time