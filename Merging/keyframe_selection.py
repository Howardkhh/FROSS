import numpy as np

class PeriodicKeyframeSelector:
    def __init__(self, interval):
        self.kf_interval = interval
        self.counter = -1

    def is_keyframe(self, *args, **kwargs):
        self.counter  = (self.counter + 1) % self.kf_interval
        if self.counter == 0:
            return True
        return False


class SpatialKeyframeSelector:
    def __init__(self, trans_thresh, rot_thresh):
        self.trans_thresh = trans_thresh
        self.rot_thresh = rot_thresh
        self.last_trans = None
        self.last_rot = None
        self.frame_since_last_kf = 0
        self.frame_count = []

    def is_keyframe(self, trans, rot, *args, **kwargs):
        if self.last_trans is None:
            self.last_trans = trans
            self.last_rot = rot
            return True

        self.frame_since_last_kf += 1
        trans_dis = np.linalg.norm(self.last_trans - trans)
        rot_angle = np.arccos((np.trace(np.dot(self.last_rot.T, rot)) - 1) / 2)

        if trans_dis > self.trans_thresh or rot_angle > self.rot_thresh:
            self.last_trans = trans
            self.last_rot = rot
            self.frame_count.append(self.frame_since_last_kf)
            self.frame_since_last_kf = 0
            return True
        return False


class DynamicKeyframeSelector:
    def __init__(self, trans_thresh, rot_thresh, iou_thresh, num_classes):
        self.trans_thresh = trans_thresh
        self.rot_thresh = rot_thresh
        self.last_trans = None
        self.last_rot = None
        self.iou_thresh = iou_thresh
        self.num_classes = num_classes
        self.last_kf_classes = np.zeros(num_classes)
        self.frame_since_last_kf = 0
        self.frame_count = []

    def _calculate_iou(self, cur_bin_counts):
        intersection = np.sum(np.minimum(self.last_kf_classes, cur_bin_counts))
        union = np.sum(np.maximum(self.last_kf_classes, cur_bin_counts))
        if union == 0:
            return 0
        return intersection / union

    def is_keyframe(self, trans, rot, cur_classes):
        if self.last_trans is None: # First frame
            self.last_trans = trans
            self.last_rot = rot
            return True

        self.frame_since_last_kf += 1
        trans_dis = np.linalg.norm(self.last_trans - trans)
        rot_angle = np.arccos((np.trace(np.dot(self.last_rot.T, rot)) - 1) / 2)
        bin_counts = np.bincount(cur_classes, minlength=self.num_classes)
        iou = self._calculate_iou(bin_counts)
        
        if trans_dis > self.trans_thresh or rot_angle > self.rot_thresh or iou < self.iou_thresh:
            self.last_trans = trans
            self.last_rot = rot
            self.last_kf_classes = bin_counts
            self.frame_count.append(self.frame_since_last_kf)
            self.frame_since_last_kf = 0
            return True
        return False