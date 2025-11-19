from math import floor
from typing import Literal
import time

import torch
import numpy as np

from utils import GaussianSG
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM_predictor = None

def init_sam2(size: Literal["tiny", "small", "base_plus", "large"] = "tiny"):
    global SAM_predictor
    if SAM_predictor is None:
        SAM_predictor = SAM2ImagePredictor.from_pretrained(f"facebook/sam2.1-hiera-{size}", hydra_overrides_extra=["model.compile_image_encoder=True"])
        print(f"Initialized SAM2 with {size} model")
    
    return SAM_predictor

# Global 3D scene graph
class GlobalSG_Gaussian:
    def __init__(self, hellinger_thershold, num_classes=20, num_rel_classes=7, visualize=False, use_sam2=False, postprocessing=False):
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.global_group = GaussianSG(num_rel_classes, hellinger_thershold)
        self.use_sam2 = use_sam2
        self.postprocessing = postprocessing
        if use_sam2:
            self.sam2 = init_sam2()
        self.visualize = visualize
        if visualize:
            self.cur_obj = []
            self.cur_rel = []

    def update(self, classes, bboxes, rels, rel_classes, depth, camera_rot, camera_trans, camera_intrinsic, img=None): # use depth camera intrinsic
        if len(classes) == 0:
            if self.visualize:
                self.cur_obj.append({"classes": self.global_group.classes.copy(), "means": self.global_group.means.copy(), "covs": self.global_group.covs.copy()})
                self.cur_rel.append(self.global_group.rels.copy())
            if self.use_sam2:
                return 0., 0., 0.

            return

        camera_rot = camera_rot[None, ...] # (1, 3, 3)
        camera_trans = camera_trans[None, :, None] # (1, 3, 1)

        if self.use_sam2:
            xyxy_bboxes = torch.cat((bboxes[:, :2] - bboxes[:, 2:] / 2, bboxes[:, :2] + bboxes[:, 2:] / 2), dim=1).int()
            bboxes = bboxes.cpu().numpy().astype(int)

            start = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.sam2.set_image(img)
                masks, _, _ = self.sam2.predict(box=xyxy_bboxes, multimask_output=False)
            if bboxes.shape[0] > 1:
                masks = masks.squeeze(1)
            sam2_time = time.time() - start

            depth_cuda = torch.tensor(depth, device="cuda")
            camera_rot_cuda = torch.tensor(camera_rot, device="cuda")
            camera_trans_cuda = torch.tensor(camera_trans, device="cuda")

            project_time, compute_mean_cov_time = 0., 0.
            mean_3d, cov_3d = [], []
            valid = np.ones((bboxes.shape[0]), dtype=bool)

            if self.postprocessing:
                # unique class IDs among objects
                unique_cls = np.unique(classes)
                class_cover = []
                for cls_id in unique_cls:
                    cls_masks = masks[classes == cls_id] 
                    cls_union = np.any(cls_masks, axis=0) 
                    class_cover.append(cls_union)
                class_cover = np.stack(class_cover, axis=0)

                # For each pixel, count how many different classes cover it
                class_cover_count = class_cover.sum(axis=0) 
                ambiguous_pixels = class_cover_count > 1 
                if ambiguous_pixels.any():
                    masks[:, ambiguous_pixels] = False
                masks = torch.from_numpy(masks).cuda()

                masks_f = masks.float().unsqueeze(1) 
                kernel_size = 7
                kernel = torch.ones((1, 1, kernel_size, kernel_size), device=masks.device)
                required = kernel_size * kernel_size
                conv = torch.nn.functional.conv2d(masks_f, kernel, padding=kernel_size // 2)
                eroded = conv == required
                masks = eroded.squeeze(1)

            else:
                masks = torch.from_numpy(masks).cuda()
                
            for mask_idx, mask in enumerate(masks):
                start = time.time()
                mask_y, mask_x = torch.nonzero(mask, as_tuple=True)
                if mask_y.shape[0] < 10:
                    valid[mask_idx] = False
                    continue
                depth_vals = depth_cuda[mask_y, mask_x].unsqueeze(-1)  # (num_points, 1)
                x = (mask_x - camera_intrinsic.cx) / camera_intrinsic.fx
                y = (mask_y - camera_intrinsic.cy) / camera_intrinsic.fy
                z = torch.ones_like(x)
            
                camera_coords = (torch.stack((x, y, z), dim=-1) * depth_vals).unsqueeze(-1)  # (num_points, 3, 1)
                coords_3d = (camera_rot_cuda @ camera_coords + camera_trans_cuda).squeeze(-1)  # (num_points, 3)
                project_time += time.time() - start
                start = time.time()
                mean_3d.append(torch.mean(coords_3d, dim=0).cpu().numpy())
                cov_3d.append((torch.cov(coords_3d.T) + 1e-6 * torch.eye(3, device=coords_3d.device)).cpu().numpy())
                compute_mean_cov_time += time.time() - start

            if len(mean_3d) == 0:
                return sam2_time, project_time, compute_mean_cov_time

            mean_3d = np.stack(mean_3d, axis=0)
            cov_3d = np.stack(cov_3d, axis=0)

            classes = classes[valid]
            bboxes = bboxes[valid]
            if len(rels) > 0:
                new_idx = np.cumsum(valid) - 1
                new_idx[~valid] = -1
                valid_edge_idx = np.logical_and(new_idx[rels[:, 0]] != -1, new_idx[rels[:, 1]] != -1)
                rel_classes = rel_classes[valid_edge_idx]
                rels = new_idx[rels[valid_edge_idx]]

        else:
            # filter out objects with invalid depth
            invalid_depth = depth[bboxes[:, 1], bboxes[:, 0]] == 0
            classes = classes[~invalid_depth]
            bboxes = bboxes[~invalid_depth] # cx, cy, w, h
            if len(rels) > 0:
                new_idx = np.cumsum(~invalid_depth) - 1
                new_idx[invalid_depth] = -1
                valid_edge_idx = np.logical_and(new_idx[rels[:, 0]] != -1, new_idx[rels[:, 1]] != -1)
                rel_classes = rel_classes[valid_edge_idx]
                rels = new_idx[rels[valid_edge_idx]]

            # project 2D to 3D

            # Eq. (2)
            x = (bboxes[:, 0] - camera_intrinsic.cx) / camera_intrinsic.fx
            y = (bboxes[:, 1] - camera_intrinsic.cy) / camera_intrinsic.fy
            z = np.ones_like(x)
            depth_val = depth[bboxes[:, 1], bboxes[:, 0], None] # (num_objects, 1)
            camera_coord = (np.stack((x, y, z), axis=-1) * depth_val)[..., None] # (num_objects, 3, 1)
            mean_3d = (camera_rot @ camera_coord + camera_trans).squeeze(-1) # (num_objects, 3)

            # Eq. (1)
            zeros = np.zeros_like(bboxes[:, 0])
            cov_2d = np.stack((bboxes[:, 2] ** 2, zeros, zeros, bboxes[:, 3] ** 2), axis=-1).reshape(-1, 2, 2) / 12

            # Eq. (3)
            J = np.array([[[camera_intrinsic.fx, 0, 0], 
                                [0, camera_intrinsic.fy, 0]]]).repeat(len(bboxes), axis=0) / depth_val[..., None]
            J[:, 0, 2] = -x * camera_intrinsic.fx / (depth_val[:, 0] ** 2)
            J[:, 1, 2] = -y * camera_intrinsic.fy / (depth_val[:, 0] ** 2)
            J_inv = np.linalg.pinv(J)

            cov_3d = J_inv @ cov_2d @ J_inv.transpose(0, 2, 1) # Eq. (5)
            cov_3d[:, 2, 2] += (cov_3d[:, 0, 0] + cov_3d[:, 1, 1]) / 2 # Eq. (6)
            cov_3d = camera_rot @ cov_3d @ camera_rot.transpose(0, 2, 1) # Eq. (7)

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

        if self.use_sam2:
            return sam2_time, project_time, compute_mean_cov_time
        return