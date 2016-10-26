from .online_sketch import OnlineSketch

import numpy as np
import numpy.linalg as ln
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.utils.extmath import safe_sparse_dot


class HybridSketch(OnlineSketch):

    """Inspired by: Streaming Anomaly Detection using Online Matrix Sketching
    """

    def _Base__clear(self):
        super()._Base__clear()
        self.freq = np.array([])

    def _Base__check(self, d):
        is_new_user, is_new_item = super()._Base__check(d)

        if is_new_item:
            self.freq = np.append(self.freq, 0)

        return is_new_user, is_new_item

    def _Base__update(self, d, is_batch_train=False):
        super()._Base__update(d, is_batch_train)
        self.freq[d['i_index']] += 1

    def _Base__recommend(self, d, target_i_indices):
        # i_mat is (n_item_context, n_item) for all possible items
        # extract only target items
        i_mat = self.i_mat[:, target_i_indices]

        n_target = len(target_i_indices)

        # u_mat will be (n_user_context, n_item) for the target user
        u = np.concatenate((d['user'], d['others']))
        u_vec = np.array([u]).T

        u_mat = sp.csr_matrix(np.repeat(u_vec, n_target, axis=1))

        # stack them into (p, n_item) matrix
        Y = sp.vstack((u_mat, i_mat))
        Y = self.proj.reduce(Y)
        Y = sp.csr_matrix(preprocessing.normalize(Y, norm='l2', axis=0))

        X = np.identity(self.k) - np.dot(self.U_r, self.U_r.T)
        A = safe_sparse_dot(X, Y, dense_output=True)

        scores = ln.norm(A, axis=0, ord=2)

        if min(scores) < 0.05:
            return self._Base__scores2recos(scores, target_i_indices)
        else:
            sorted_indices = np.argsort(self.freq[target_i_indices])[::-1]
            return target_i_indices[sorted_indices], self.freq[target_i_indices][sorted_indices]
