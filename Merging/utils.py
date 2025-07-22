import numpy as np

# 3D scene graph with Gaussian representation
class GaussianSG:
    def __init__(self, num_rel_class, merge_threshold):
        initial_size = 1000
        self._growth_factor = 2
        self._max_size = initial_size
        self._valid_mask = np.zeros((initial_size), dtype=bool)
        self._classes = np.ndarray((initial_size), dtype=int)
        self._means = np.ndarray((initial_size, 3), dtype=float)
        self._covs = np.ndarray((initial_size, 3, 3), dtype=float)
        self._rels = np.zeros((initial_size, initial_size, num_rel_class), dtype=int)
        self._pcd = [None] * initial_size
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

        self._classes = np.concatenate((self._classes, np.ndarray((add_size), dtype=int)))
        self._means = np.concatenate((self._means, np.ndarray((add_size, 3), dtype=float)))
        self._covs = np.concatenate((self._covs, np.ndarray((add_size, 3, 3), dtype=float)))
        self._rels = np.concatenate((self._rels, np.zeros((add_size, self._max_size, self.num_rel_class), dtype=int)), axis=0)
        self._rels = np.concatenate((self._rels, np.zeros((new_size, add_size, self.num_rel_class), dtype=int)), axis=1)
        self._pcd.extend([None] * add_size)
        self._valid_mask = np.concatenate((self._valid_mask, np.zeros((add_size), dtype=bool)))
        self._max_size = new_size

    def add(self, new_classes, new_means, new_covs, new_rels, new_rel_classes, new_pcds):
        add_size = len(new_classes)
        self._expand_if_needed(add_size)
        avail_indices = np.nonzero(~self._valid_mask)[0][:add_size]
        self._classes[avail_indices] = new_classes
        self._means[avail_indices] = new_means
        self._covs[avail_indices] = new_covs
        for idx, avail_idx in enumerate(avail_indices):
            self._pcd[avail_idx] = new_pcds[idx]
        if len(new_rels) > 0:
            new_rels = avail_indices[new_rels]
            self._rels[new_rels[:, 0], new_rels[:, 1], new_rel_classes] += 1
        
        self._valid_mask[avail_indices] = True

        return avail_indices # idx to update
    
    def merge(self, update_idx):
        update_idx = update_idx.tolist()
        while update_idx:
            idx = update_idx.pop()
            same_class = self._classes == self._classes[idx]
            same_class[idx] = False # exclude the current Gaussian itself
            same_class = same_class & self._valid_mask
            same_class = np.nonzero(same_class)[0]
            if np.count_nonzero(same_class) == 0:
                continue
            mean1 = self._means[idx]
            cov1 = self._covs[idx]
            hell_dist = self._batched_hellinger_distance(mean1, cov1, self._means[same_class], self._covs[same_class])
            min_idx = np.argmin(hell_dist)
            if hell_dist[min_idx] < self.merge_threshold:
                merge_idx = same_class[min_idx]
                if merge_idx not in update_idx: update_idx.append(merge_idx)
                assert self._valid_mask[merge_idx] and self._valid_mask[idx]
                self._merge_gaussians(idx, merge_idx)

    def _batched_hellinger_distance(self, mean1, cov1, mean2, cov2):
        """
        Calculate the Hellinger distance between one and many Gaussian distributions
        """
        assert len(mean1.shape) == 1 and len(mean2.shape) == 2
        assert len(cov1.shape) == 2 and len(cov2.shape) == 3
        assert mean1.shape[0] == mean2.shape[1] == 3
        assert cov1.shape[0] == cov1.shape[1] == cov2.shape[1] == cov2.shape[2] == 3
        mean1 = mean1[None, :, None]
        mean2 = mean2[..., None]
        cov1 = cov1[None, ...]
        mean_diff = mean1 - mean2
        cov_mean = (cov1 + cov2) / 2
        cov_mean_inv = np.linalg.inv(cov_mean)
        det_cov_mean = np.linalg.det(cov_mean)
        B_D = (0.125 * mean_diff.transpose(0, 2, 1) @ cov_mean_inv @ mean_diff).flatten() \
            + 0.5 * np.log(det_cov_mean / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
        return np.sqrt(1 - np.exp(-B_D))

    def _merge_gaussians(self, idx1, idx2): # merge idx1 to idx2
        mean1, cov1 = self._means[idx1], self._covs[idx1]
        mean2, cov2 = self._means[idx2], self._covs[idx2]
        assert len(mean1.shape) == len(mean2.shape) == 1
        assert len(cov1.shape) == len(cov2.shape) == 2
        assert mean1.shape[0] == mean2.shape[0] == 3
        assert cov1.shape[1] == cov1.shape[0] == cov2.shape[1] == cov2.shape[0] == 3
        num_pnts1, num_pnts2 = len(self._pcd[idx1]), len(self._pcd[idx2])
        if num_pnts1 == 0 and num_pnts2 == 0: # too small so that they don't have pcd
            num_pnts1 = num_pnts2 = 1
        total_pnts = num_pnts1 + num_pnts2
        mean_diff = mean1 - mean2
        self._means[idx2] = (num_pnts1 * mean1 + num_pnts2 * mean2) / total_pnts
        self._covs[idx2] = (num_pnts1 * cov1 + num_pnts2 * cov2) / total_pnts + num_pnts1 * num_pnts2 * np.outer(mean_diff, mean_diff) / total_pnts**2
        self._rels[idx2, self._valid_mask] += self._rels[idx1, self._valid_mask]
        self._rels[self._valid_mask, idx2] += self._rels[self._valid_mask, idx1]
        self._pcd[idx2] = np.concatenate((self._pcd[idx1], self._pcd[idx2]))
        self._classes[idx1] = -9999999
        self._means[idx1] = np.nan
        self._covs[idx1] = np.nan
        self._rels[idx1, self._valid_mask] = 0
        self._rels[self._valid_mask, idx1] = 0
        self._pcd[idx1] = None
        self._valid_mask[idx1] = False
