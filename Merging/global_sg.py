from math import floor

import numpy as np
from color_histogram.core.hist_3d import Hist3D
import webcolors
import open3d as o3d

from utils import GaussianSG, KimSG

# Global 3D scene graph
class GlobalSG_Gaussian:
    def __init__(self, hellinger_thershold, num_classes=20, num_rel_classes=7):
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.global_group = GaussianSG(num_rel_classes, hellinger_thershold)

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
        
        return
    
# Below is partially modified from 3-D Scene Graph, Kim et al. 2019 (https://github.com/Uehwan/3-D-Scene-Graph)
class find_objects_class_and_color(object):
    def __init__(self):
        self.power = 2

    def get_class_string(self, class_index, score, dataset):
        class_text = dataset[class_index] if dataset is not None else \
            'id{:d}'.format(class_index)
        return class_text + ' {:0.2f}'.format(score).lstrip('0')

    def closest_colour(self, requested_colour):
        min_colours = {}
        for name in webcolors.names("css3"):
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            rd = (r_c - requested_colour[0]) ** self.power
            gd = (g_c - requested_colour[1]) ** self.power
            bd = (b_c - requested_colour[2]) ** self.power
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def get_colour_name(self, requested_colour):
        try:
            closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
        except ValueError as e:
            closest_name = FOCC.closest_colour(requested_colour)
            actual_name = None
        return actual_name, closest_name

class resampling_boundingbox_size(object):
    def __init__(self):
        self.range = 10.0
        self.mean_k = 10
        self.thres = 1.0

    def isNotNoisyPoint(self, point):
        return -self.range< point[0]<self.range and -self.range<point[1]<self.range and -self.range<point[2]<self.range

    def outlier_filter(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=self.mean_k, std_ratio=self.thres)
        return np.asarray(cl.points)

    def make_window_size(self, width, height, obj_boxes):
        x1, y1 = obj_boxes[0] - obj_boxes[2] / 2, obj_boxes[1] - obj_boxes[3] / 2
        if( width<30):
            range_x_min = int(x1) + int(width*3./10.) 
            range_x_max = int(x1) + int(width*7./10.)
        elif(width < 60):
            range_x_min = int(x1) + int(width*8./20.)
            range_x_max = int(x1) + int(width*12./20.)
        else:
            range_x_min = int(x1) + int(width*12./30.)
            range_x_max = int(x1) + int(width*18./30.)
            
        if (height < 30):
            range_y_min = int(y1) + int(height*3./10.)
            range_y_max = int(y1) + int(height*7./10.)
        elif (height < 60):
            range_y_min = int(y1) + int(height*8./20.)
            range_y_max = int(y1) + int(height*12./20.)
        else:
            range_y_min = int(y1) + int(height*12./30.)
            range_y_max = int(y1) + int(height*18./30.)

        return range_x_min, range_x_max, range_y_min, range_y_max

FOCC = find_objects_class_and_color()
RBS = resampling_boundingbox_size()

class GlobalSG_Kim:
    def __init__(self, hellinger_thershold, num_classes=20, num_rel_classes=7):
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.global_group = KimSG(num_classes, num_rel_classes, hellinger_thershold)

    def get_histograms(self, img):
            hist3D = Hist3D(img, num_bins=8, color_space='rgb')
            densities = hist3D.colorDensities()
            order = densities.argsort()[::-1]
            densities = densities[order]
            colors = (255*hist3D.rgbColors()[order]).astype(int)
            color_hist = []
            for density, color in zip(densities[:4],colors[:4]):
                actual_name, closest_name = FOCC.get_colour_name(color.tolist())
                if (actual_name == None):
                    color_hist.append([density, closest_name])
                else:
                    color_hist.append([density, actual_name])

            return color_hist

    def update(self, classes, bboxes, rels, rel_classes, depth, camera_rot, camera_trans, camera_intrinsic, img): # use depth camera intrinsic
        if len(classes) == 0:
            return

        camera_rot = camera_rot # (3, 3)
        camera_trans = camera_trans[:, None] # (3, 1)

        full_boxes = []
        histograms = []
        points_3d = []
        valid_points = np.ones(len(bboxes), dtype=bool)
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            x1, y1 = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2
            x2, y2 = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            x1, x2, y1, y2 = max(0, x1), min(img.shape[1], x2), max(0, y1), min(img.shape[0], y2)
            if x2-x1 <= 0 or y2-y1 <= 0:
                valid_points[i] = False
                continue
            img_crop = img[y1:y2, x1:x2]
            full_boxes.append(img_crop)
            histograms.append(self.get_histograms(img_crop))
            '''2. Get Center Patch '''
            # Define bounding box info
            width = int(bboxes[i][2])
            height = int(bboxes[i][3])
            # using belows to find mean and variance of each bounding boxes
            # pop 1/5 size window_box from object bounding boxes 
            range_x_min, range_x_max, range_y_min, range_y_max = RBS.make_window_size(width, height, bboxes[i])

            '''3. Get 3D positions of the Centor Patch'''
            window_3d_pts = []
            for pt_x in range(range_x_min, range_x_max):
                for pt_y in range(range_y_min, range_y_max):
                    x = (pt_x - camera_intrinsic.cx) / camera_intrinsic.fx
                    y = (pt_y - camera_intrinsic.cy) / camera_intrinsic.fy
                    z = np.ones_like(x)
                    depth_val = depth[pt_y, pt_x] # (1)
                    camera_coord = (np.stack((x, y, z), axis=-1) * depth_val)[..., None] # (3, 1)
                    mean_3d = (camera_rot @ camera_coord + camera_trans).squeeze(-1) # (3)
        
                    if RBS.isNotNoisyPoint(mean_3d):
                        # save several points in window_box to calculate mean and variance
                        window_3d_pts.append([mean_3d[0], mean_3d[1], mean_3d[2]])
            if len(window_3d_pts) <= 10:
                del full_boxes[-1]
                del histograms[-1]
                valid_points[i] = False
                continue
            window_3d_pts = RBS.outlier_filter(window_3d_pts)
            points_3d.append(window_3d_pts)

        # Add local 3D SG to global 3D SG
        classes = classes[valid_points]
        bboxes = bboxes[valid_points]
        if len(rels) > 0:
            new_idx = np.cumsum(valid_points) - 1
            new_idx[~valid_points] = -1
            valid_edge_idx = np.logical_and(new_idx[rels[:, 0]] != -1, new_idx[rels[:, 1]] != -1)
            rel_classes = rel_classes[valid_edge_idx]
            rels = new_idx[rels[valid_edge_idx]]
        update_idx = self.global_group.add(classes, rels, rel_classes, points_3d, histograms)

        # Calculate the Hellinger distance and merge objects
        self.global_group.merge(update_idx)
        
        return