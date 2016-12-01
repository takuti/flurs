from flurs.recommender import feature_recommender

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot

from logging import getLogger, StreamHandler, Formatter, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setFormatter(Formatter('[%(process)d] %(message)s'))
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


class IncrementalFMs(feature_recommender.FeatureRecommender):

    """Incremental Factorization Machines
    """

    def __init__(
            self, contexts, is_static=False, k=40, l2_reg_w0=2., l2_reg_w=8., l2_reg_V=16., learn_rate=.004):
        super().__init__()

        self.contexts = contexts
        self.p = np.sum(list(contexts.values()))

        self.is_static = is_static

        self.k = k
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = np.ones(k) * l2_reg_V
        self.learn_rate = learn_rate

        self.init_model()

    def init_model(self):
        self.i_mat = sp.csr_matrix([])

        # initial parameters
        self.w0 = 0.
        self.w = np.zeros(self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))

        # to keep the last parameters for adaptive regularization
        self.prev_w0 = self.w0
        self.prev_w = self.w.copy()
        self.prev_V = self.V.copy()

    def add_user(self, u, feature):
        super().add_user(u, feature)

        n_user = self.n_user - 1

        # insert new user's row for the parameters
        self.w = np.concatenate((self.w[:n_user], np.array([0.]), self.w[n_user:]))
        self.prev_w = np.concatenate((self.prev_w[:n_user], np.array([0.]), self.prev_w[n_user:]))

        rand_row = np.random.normal(0., 0.1, (1, self.k))
        self.V = np.concatenate((self.V[:n_user], rand_row, self.V[n_user:]))
        self.prev_V = np.concatenate((self.prev_V[:n_user], rand_row, self.prev_V[n_user:]))

        self.p += 1

    def add_item(self, i, feature):
        super().add_item(i, feature)

        n_item = self.n_item - 1

        # insert new item's row for the parameters
        h = self.n_user + n_item
        self.w = np.concatenate((self.w[:h], np.array([0.]), self.w[h:]))
        self.prev_w = np.concatenate((self.prev_w[:h], np.array([0.]), self.prev_w[h:]))

        rand_row = np.random.normal(0., 0.1, (1, self.k))
        self.V = np.concatenate((self.V[:h], rand_row, self.V[h:]))
        self.prev_V = np.concatenate((self.prev_V[:h], rand_row, self.prev_V[h:]))

        # update the item matrix for all items
        i_vec = np.concatenate((np.zeros(n_item + 1), feature))
        i_vec[i] = 1.
        sp_i_vec = sp.csr_matrix(np.array([i_vec]).T)

        if self.i_mat.size == 0:
            self.i_mat = sp_i_vec
        else:
            self.i_mat = sp.vstack((self.i_mat[:n_item], np.zeros((1, n_item)), self.i_mat[n_item:]))
            self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, sp_i_vec)))

        self.p += 1

    def update(self, u, i, r, context, is_batch_train=False):
        # static baseline; w/o updating the model
        if not is_batch_train and self.is_static:
            return

        # create user/item ID vector
        x = np.zeros(self.n_user + self.n_item)
        x[u] = x[self.n_user + i] = 1.

        # append contextual variables
        x = np.concatenate((x, self.users[u]['feature'], context, self.items[i]['feature']))

        x_vec = np.array([x]).T  # p x 1
        interaction = np.sum(np.dot(self.V.T, x_vec) ** 2 - np.dot(self.V.T ** 2, x_vec ** 2)) / 2.
        pred = self.w0 + np.inner(self.w, x) + interaction

        # compute current error
        err = r - pred

        # update regularization parameters
        coeff = 4. * self.learn_rate * err * self.learn_rate

        self.l2_reg_w0 = max(0., self.l2_reg_w0 + coeff * self.prev_w0)
        self.l2_reg_w = max(0., self.l2_reg_w + coeff * np.inner(x, self.prev_w))

        if self.l2_reg_w0 == 0. or self.l2_reg_w == 0.:
            logger.debug('[warn] reg_w0 and/or reg_w are fallen in 0.0')

        dot_v = np.dot(x_vec.T, self.V).reshape((self.k,))  # (k, )
        dot_prev_v = np.dot(x_vec.T, self.prev_V).reshape((self.k,))  # (k, )
        s_duplicated = np.dot((x_vec.T ** 2), self.V * self.prev_V).reshape((self.k,))  # (k, )
        self.l2_rev_V = np.maximum(np.zeros(self.k), self.l2_reg_V + coeff * (dot_v * dot_prev_v - s_duplicated))

        # update w0 with keeping the previous value
        self.prev_w0 = self.w0
        self.w0 = self.w0 + 2. * self.learn_rate * (err * 1. - self.l2_reg_w0 * self.w0)

        # keep the previous w for auto-parameter optimization
        self.prev_w = np.empty_like(self.w)
        self.prev_w[:] = self.w

        # keep the previous V
        self.prev_V = np.empty_like(self.V)
        self.prev_V[:] = self.V

        # update w and V
        prod = np.dot(np.array([x]), self.prev_V)  # (1, p) and (p, k) => (1, k)
        for pi in range(self.p):
            if x[pi] == 0.:
                continue

            self.w[pi] = self.prev_w[pi] + 2. * self.learn_rate * (err * x[pi] - self.l2_reg_w * self.prev_w[pi])

            g = err * x[pi] * (prod - x[pi] * self.prev_V[pi])
            self.V[pi] = self.prev_V[pi] + 2. * self.learn_rate * (g - self.l2_reg_V * self.prev_V[pi])

    def recommend(self, u, target_i_indices, context):
        # i_mat is (n_item_context, n_item) for all possible items
        # extract only target items
        i_mat = self.i_mat[:, target_i_indices]

        n_target = len(target_i_indices)

        # u_mat will be (n_user + n_user_context, n_item) for the target user
        u_vec = np.concatenate((np.zeros(self.n_user), self.users[u]['feature'], context))
        u_vec[u] = 1.
        u_vec = np.array([u_vec]).T
        u_mat = sp.csr_matrix(np.repeat(u_vec, n_target, axis=1))

        # stack them into (p, n_item) matrix
        # rows are ordered by [user ID - item ID - user context - others - item context]
        mat = sp.vstack((u_mat[:self.n_user], i_mat[:self.n_item], u_mat[self.n_user:], i_mat[self.n_item:]))

        # Matrix A and B should be dense (numpy array; rather than scipy CSR matrix) because V is dense.
        V = sp.csr_matrix(self.V)
        A = safe_sparse_dot(V.T, mat)
        A.data[:] = A.data ** 2

        sq_mat = mat.copy()
        sq_mat.data[:] = sq_mat.data ** 2
        sq_V = V.copy()
        sq_V.data[:] = sq_V.data ** 2
        B = safe_sparse_dot(sq_V.T, sq_mat)

        interaction = (A - B).sum(axis=0)
        interaction /= 2.  # (1, n_item); numpy matrix form

        pred = self.w0 + safe_sparse_dot(self.w, mat, dense_output=True) + interaction

        scores = np.abs(1. - np.ravel(pred))

        return self.scores2recos(scores, target_i_indices)
