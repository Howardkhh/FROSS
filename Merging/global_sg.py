from math import floor

import numpy as np

from utils import GaussianSG

class GlobalSG_Gaussian:
    def __init__(self, hellinger_thershold, num_classes=20, num_rel_classes=7, draw_colors=None):
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.global_group = GaussianSG(num_rel_classes, hellinger_thershold)
        self.draw_colors = draw_colors # * 255 if draw_colors is not None else None

    def update(self, classes, bboxes, rels, rel_classes, depth, camera_rot, camera_trans, camera_intrinsic): # use depth camera intrinsic
        if len(classes) == 0:
            return

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

        # project 2D to 3D
        x = (bboxes[:, 0] - camera_intrinsic.cx) / camera_intrinsic.fx
        y = (bboxes[:, 1] - camera_intrinsic.cy) / camera_intrinsic.fy
        z = np.ones_like(x)
        depth_val = depth[bboxes[:, 1], bboxes[:, 0], None] # (num_objects, 1)
        camera_coord = (np.stack((x, y, z), axis=-1) * depth_val)[..., None] # (num_objects, 3, 1)
        mean_3d = (camera_rot @ camera_coord + camera_trans).squeeze(-1) # (num_objects, 3)

        zeros = np.zeros_like(bboxes[:, 0])
        cov_2d = np.stack((bboxes[:, 2] ** 2, zeros, zeros, bboxes[:, 3] ** 2), axis=-1).reshape(-1, 2, 2) / 12

        J = np.array([[[camera_intrinsic.fx, 0, 0], 
                             [0, camera_intrinsic.fy, 0]]]).repeat(len(bboxes), axis=0) / depth_val[..., None]
        J[:, 0, 2] = -x * camera_intrinsic.fx / (depth_val[:, 0] ** 2)
        J[:, 1, 2] = -y * camera_intrinsic.fy / (depth_val[:, 0] ** 2)
        J_inv = np.linalg.pinv(J)

        cov_3d = J_inv @ cov_2d @ J_inv.transpose(0, 2, 1)
        cov_3d[:, 2, 2] += (cov_3d[:, 0, 0] + cov_3d[:, 1, 1]) / 2
        cov_3d = camera_rot @ cov_3d @ camera_rot.transpose(0, 2, 1)

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

        # add all 2D objects to global 3DSG
        update_idx = self.global_group.add(classes, mean_3d, cov_3d, rels, rel_classes, pcds)

        self.global_group.merge(update_idx)
        
        return