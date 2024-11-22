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


class DynamicKeyframeSelector:
    def __init__(self, iou_thresh, max_interval, num_classes):
        self.iou_thresh = iou_thresh
        self.max_interval = max_interval
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

    def is_keyframe(self, cur_classes):
        self.frame_since_last_kf += 1
        bin_counts = np.bincount(cur_classes, minlength=self.num_classes)
        iou = self._calculate_iou(bin_counts)
        if iou < self.iou_thresh or self.frame_since_last_kf >= self.max_interval:
            self.last_kf_classes = bin_counts
            self.frame_count.append(self.frame_since_last_kf)
            self.frame_since_last_kf = 0
            return True
        return False