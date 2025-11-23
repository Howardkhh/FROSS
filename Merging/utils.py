import numpy as np

# 3D scene graph with Gaussian representation
class GaussianSG:
    def __init__(self, num_obj_class, num_rel_class, merge_threshold, dist_threshold, classes_dist_method):
        initial_size = 1000
        self._growth_factor = 2
        self._max_size = initial_size
        self._valid_mask = np.zeros((initial_size), dtype=bool)
        # self._classes = np.ndarray((initial_size), dtype=int)
        self._means = np.ndarray((initial_size, 3), dtype=float)
        self._covs = np.ndarray((initial_size, 3, 3), dtype=float)
        self._rels = np.zeros((initial_size, initial_size, num_rel_class), dtype=int)
        self._pcd = [None] * initial_size
        self.num_rel_class = num_rel_class
        self.merge_threshold = merge_threshold
        self.classes_dist_method = classes_dist_method

        # introduce class prob distribution distance in merging criteria
        self.num_obj_class = num_obj_class
        self._classes = np.ndarray((initial_size, num_obj_class), dtype=float)
        self.classes_dist_threshold = dist_threshold

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
        # self._classes = np.concatenate((self._classes, np.ndarray((add_size), dtype=int)))
        self._classes = np.concatenate((self._classes, np.ndarray((add_size, self.num_obj_class), dtype=float)))
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
        # print(f"classes to add: {new_classes.shape}, \
        #     self._classes shape: {self._classes.shape}, \
        #     means to add: {new_means.shape}, \
        #     covs to add: {new_covs.shape}, \
        #     rels to add: {new_rels.shape}, \
        #     rel_classes to add: {new_rel_classes.shape}"
        # )
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
    
    # update for FROSS v2: merge with different criteria
    def merge(self, update_idx):
        update_idx = update_idx.tolist()
        while update_idx:
            idx = update_idx.pop()
            # same_class = self._classes == self._classes[idx]
            # same_class[idx] = False # exclude the current Gaussian itself
            # same_class = same_class & self._valid_mask
            # same_class = np.nonzero(same_class)[0]
            # if np.count_nonzero(same_class) == 0:
            #     continue
            if not self._valid_mask[idx]:
                continue
            all_valid_indices = np.nonzero(self._valid_mask)[0]
            other_indices = all_valid_indices[all_valid_indices != idx]
            if len(other_indices) == 0:
                continue
            # check class distribution distance
            class_dists = self._batched_distribution_distance(self._classes[idx], self._classes[other_indices], metric=self.classes_dist_method)
            same_class = class_dists < self.classes_dist_threshold
            same_class_indices = other_indices[same_class]
            if len(same_class_indices) == 0:
                continue
            
            mean1 = self._means[idx]
            cov1 = self._covs[idx]
            hell_dist = self._batched_hellinger_distance(mean1, cov1, self._means[same_class_indices], self._covs[same_class_indices])
            min_idx = np.argmin(hell_dist)
            if hell_dist[min_idx] < self.merge_threshold:
                merge_idx = same_class_indices[min_idx]
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
        # distribution distance check
        dist1 = self._classes[idx1]
        dist2 = self._classes[idx2]
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
        
        # update class distribution
        self._classes[idx2] = (num_pnts1 * dist1 + num_pnts2 * dist2) / total_pnts
        self._classes[idx1] = np.nan
        #
        self._means[idx1] = np.nan
        self._covs[idx1] = np.nan
        self._rels[idx1, self._valid_mask] = 0
        self._rels[self._valid_mask, idx1] = 0
        self._pcd[idx1] = None
        self._valid_mask[idx1] = False

    def _batched_distribution_distance(self, p, q_batch, metric='kl'):
        """
        Compute distance between one probability distribution p and a batch of distributions q_batch.
        p is a 1D array. q_batch is a 2D array (N, K).
        Returns an array of distances (N,).
        """
        # Ensure p and q_batch are valid probability distributions
        p = np.clip(p, 1e-12, 1.0)
        q_batch = np.clip(q_batch, 1e-12, 1.0)
        
        if metric == 'hellinger':
            return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q_batch)) ** 2, axis=1))
        elif metric == 'kl':
            return np.sum(p * np.log(p / q_batch), axis=1)
        elif metric == 'js':
            m = 0.5 * (p + q_batch) # m shape is (N, K)
            kl_p_m = np.sum(p * np.log(p / m), axis=1) # shape (N,)
            kl_q_m = np.sum(q_batch * np.log(q_batch / m), axis=1) # shape (N,)
            return 0.5 * kl_p_m + 0.5 * kl_q_m
        elif metric == 'l2':
            return np.sqrt(np.sum((p - q_batch) ** 2, axis=1))
        elif metric == 'dot_product':
            return 1.0 - np.sum(p * q_batch, axis=1)
        elif metric == "top_class":
            p_top = np.argmax(p)
            q_top = np.argmax(q_batch, axis=1)
            return (p_top != q_top).astype(float)
        else:
            raise ValueError(f"Unknown metric: {metric}")