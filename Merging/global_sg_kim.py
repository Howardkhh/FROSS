# Partially modified from 3-D Scene Graph, Kim et al. 2019 (https://github.com/Uehwan/3-D-Scene-Graph)

from math import floor

import numpy as np
# pip install webcolors==24.11.1 git+https://github.com/chickenbestlover/ColorHistogram.git
import webcolors
from color_histogram.core.hist_3d import Hist3D
import open3d as o3d

class KimSG:
    def __init__(self, num_obj_class, num_rel_class, merge_threshold):
        initial_size = 1000
        self._growth_factor = 2
        self._max_size = initial_size
        self._valid_mask = np.zeros((initial_size), dtype=bool)
        self._classes = np.ndarray((initial_size, num_obj_class), dtype=float)
        self._means = np.ndarray((initial_size, 3), dtype=float)
        self._covs = np.ndarray((initial_size, 3, 3), dtype=float)
        self._rels = np.zeros((initial_size, initial_size, num_rel_class), dtype=int)
        self._pcd = [None] * initial_size
        self._histogram = [None] * initial_size
        self._num_obj_class = num_obj_class
        self.num_rel_class = num_rel_class
        self.merge_threshold = merge_threshold

    @property
    def valid_size(self):
        return np.count_nonzero(self._valid_mask)
    @property
    def classes(self):
        return self._classes[self._valid_mask]
    @property
    def means(self):
        return self._means[self._valid_mask]
    @property
    def covs(self):
        return self._covs[self._valid_mask]
    @property
    def rels(self):
        return self._rels[self._valid_mask][:, self._valid_mask]
    @property
    def pcd(self):
        return [self._pcd[i] for i in range(len(self._pcd)) if self._valid_mask[i]]

    def _expand_if_needed(self, add_size):
        if self.valid_size + add_size <= self._max_size:
            return
        new_size = self._max_size * self._growth_factor + add_size
        add_size = new_size - self._max_size

        self._classes = np.concatenate((self._classes, np.ndarray((add_size, self._num_obj_class), dtype=float)), axis=0)
        self._means = np.concatenate((self._means, np.ndarray((add_size, 3), dtype=float)))
        self._covs = np.concatenate((self._covs, np.ndarray((add_size, 3, 3), dtype=float)))
        self._rels = np.concatenate((self._rels, np.zeros((add_size, self._max_size, self.num_rel_class), dtype=int)), axis=0)
        self._rels = np.concatenate((self._rels, np.zeros((new_size, add_size, self.num_rel_class), dtype=int)), axis=1)
        self._pcd.extend([None] * add_size)
        self._histogram.extend([None] * add_size)
        self._valid_mask = np.concatenate((self._valid_mask, np.zeros((add_size), dtype=bool)))
        self._max_size = new_size

    def add(self, new_classes, new_rels, new_rel_classes, new_pcds, new_histograms):
        add_size = len(new_classes)
        self._expand_if_needed(add_size)
        avail_indices = np.nonzero(~self._valid_mask)[0][:add_size]
        self._classes[avail_indices] = new_classes
        new_means = np.ndarray((add_size, 3), dtype=float)
        new_covs = np.ndarray((add_size, 3, 3), dtype=float)
        for i in range(add_size):
            new_means[i] = np.mean(new_pcds[i], axis=0)
            new_covs[i] = np.cov(new_pcds[i], rowvar=False)
        self._means[avail_indices] = new_means
        self._covs[avail_indices] = new_covs
        for idx, avail_idx in enumerate(avail_indices):
            self._pcd[avail_idx] = new_pcds[idx]
            self._histogram[avail_idx] = new_histograms[idx]
        if len(new_rels) > 0:
            new_rels = avail_indices[new_rels]
            self._rels[new_rels[:, 0], new_rels[:, 1], new_rel_classes] += 1

        self._valid_mask[avail_indices] = True

        return avail_indices # idx to update

    def merge(self, update_idx):
        update_idx = update_idx.tolist()
        while update_idx:
            idx = update_idx.pop()
            if np.count_nonzero(self._valid_mask) == 1:
                continue
            similarity = self.node_similarity(idx)
            max_idx = np.argmax(similarity)
            if similarity[max_idx] > self.merge_threshold:
                merge_idx = np.nonzero(self._valid_mask)[0][max_idx]
                if merge_idx not in update_idx: update_idx.append(merge_idx)
                assert self._valid_mask[merge_idx] and self._valid_mask[idx]
                self._merge_gaussians(idx, merge_idx)

    def compare_class(self, cls1, cls2):
        top3_cls1 = (-cls1).argsort()[:3] # choose top 3 index
        top3_cls2 = (-cls2).argsort()[:3] # choose top 3 index
        cls_inter = np.intersect1d(top3_cls1, top3_cls2).size
        if cls_inter == 0:
            cls_score = 0
        elif cls_inter == 1:
            cls_score = 0.8
        elif cls_inter == 2:
            cls_score = 0.9
        elif cls_inter == 3:
            cls_score = 1.0
        return cls_score

    def compare_position(self, mean1, mean2):
        Z_x = (mean1[0]-mean2[0])/5
        Z_y = (mean1[1]-mean2[1])/5
        Z_z = (mean1[2]-mean2[2])/5
        x_check = -0.112 < Z_x < 0.112
        y_check = -0.112 < Z_y < 0.112
        z_check = -0.112 < Z_z < 0.112

        if (x_check):
            I_x = 1.0
        else:
            I_x = 0.112 / np.abs(Z_x)
        if (y_check):
            I_y = 1.0
        else:
            I_y = 0.112 / np.abs(Z_y)
        if (z_check):
            I_z = 1.0
        else:
            I_z = 0.112 / np.abs(Z_z)

        score = (I_x/3.0) + (I_y/3.0) + (I_z/3.0)

        return score

    def compare_color(self, curr_hist, prev_hist):
        curr_rgb = webcolors.name_to_rgb(curr_hist[0][1])
        prev_rgb = webcolors.name_to_rgb(prev_hist[0][1])
        dist = np.sqrt(np.sum(np.power(np.subtract(curr_rgb, prev_rgb),2))) / (255*np.sqrt(3))
        score = 1-dist
        return score

    def node_similarity(self, idx):
        mean = self._means[idx]
        cls = self._classes[idx]
        hist = self._histogram[idx]
        scores = []
        for j in np.nonzero(self._valid_mask)[0]:
            if j == idx:
                scores.append(-1)
                continue
            mean2 = self._means[j]
            cls2 = self._classes[j]
            hist2 = self._histogram[j]

            cls_score = self.compare_class(cls, cls2)
            pos_score = self.compare_position(mean, mean2)
            col_score = self.compare_color(hist, hist2)
            total_score = 0.5 * cls_score + 0.4 * pos_score + 0.1 * col_score
            scores.append(total_score)
        scores = np.array(scores)
        return scores

    def _merge_gaussians(self, idx1, idx2): # merge idx1 to idx2
        mean1, cov1 = self._means[idx1], self._covs[idx1]
        mean2, cov2 = self._means[idx2], self._covs[idx2]
        assert len(mean1.shape) == len(mean2.shape) == 1
        assert len(cov1.shape) == len(cov2.shape) == 2
        assert mean1.shape[0] == mean2.shape[0] == 3
        assert cov1.shape[1] == cov1.shape[0] == cov2.shape[1] == cov2.shape[0] == 3
        num_pnts1, num_pnts2 = len(self._pcd[idx1]), len(self._pcd[idx2])
        if num_pnts1 == 0 and num_pnts2 == 0:
            num_pnts1 = num_pnts2 = 1 # too small so that they don't have pcd
        total_pnts = num_pnts1 + num_pnts2
        mean_diff = mean1 - mean2
        self._means[idx2] = (num_pnts1 * mean1 + num_pnts2 * mean2) / total_pnts
        self._covs[idx2] = (num_pnts1 * cov1 + num_pnts2 * cov2) / total_pnts + num_pnts1 * num_pnts2 * np.outer(mean_diff, mean_diff) / total_pnts**2
        self._rels[idx2, self._valid_mask] += self._rels[idx1, self._valid_mask]
        self._rels[self._valid_mask, idx2] += self._rels[self._valid_mask, idx1]
        self._pcd[idx2] = np.concatenate((self._pcd[idx1], self._pcd[idx2]))
        self._classes[idx2] = self._classes[idx2] + self._classes[idx1]
        self._histogram[idx2] = self._histogram[idx1]
        self._classes[idx1] = np.ones((self._num_obj_class), dtype=float) * -9999999
        self._means[idx1] = np.nan
        self._covs[idx1] = np.nan
        self._rels[idx1, self._valid_mask] = 0
        self._rels[self._valid_mask, idx1] = 0
        self._pcd[idx1] = None
        self._histogram[idx1] = None
        self._valid_mask[idx1] = False


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