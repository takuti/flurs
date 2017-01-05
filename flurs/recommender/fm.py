from flurs.base import FeatureRecommenderMixin
from flurs.model.fm import FactorizationMachine

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


class FMRecommender(FactorizationMachine, FeatureRecommenderMixin):

    def init_recommender(self, is_static=False):
        super().init_recommender()
        self.is_static = is_static

    def add_user(self, user):
        super().add_user(user)

        n_user = self.n_user - 1

        # insert new user's row for the parameters
        self.w = np.concatenate((self.w[:n_user],
                                 np.array([0.]),
                                 self.w[n_user:]))

        self.prev_w = np.concatenate((self.prev_w[:n_user],
                                      np.array([0.]),
                                      self.prev_w[n_user:]))

        rand_row = np.random.normal(0., 0.1, (1, self.k))

        self.V = np.concatenate((self.V[:n_user],
                                 rand_row,
                                 self.V[n_user:]))

        self.prev_V = np.concatenate((self.prev_V[:n_user],
                                      rand_row,
                                      self.prev_V[n_user:]))

        self.p += 1

    def add_item(self, item):
        super().add_item(item)

        n_item = self.n_item - 1

        # insert new item's row for the parameters
        h = self.n_user + n_item

        self.w = np.concatenate((self.w[:h],
                                 np.array([0.]),
                                 self.w[h:]))

        self.prev_w = np.concatenate((self.prev_w[:h],
                                      np.array([0.]),
                                      self.prev_w[h:]))

        rand_row = np.random.normal(0., 0.1, (1, self.k))

        self.V = np.concatenate((self.V[:h],
                                 rand_row,
                                 self.V[h:]))

        self.prev_V = np.concatenate((self.prev_V[:h],
                                      rand_row,
                                      self.prev_V[h:]))

        # update the item matrix for all items
        i_vec = np.concatenate((np.zeros(n_item + 1), item.feature))
        i_vec[item.index] = 1.
        sp_i_vec = sp.csr_matrix(np.array([i_vec]).T)

        if self.i_mat.size == 0:
            self.i_mat = sp_i_vec
        else:
            self.i_mat = sp.vstack((self.i_mat[:n_item],
                                    np.zeros((1, n_item)),
                                    self.i_mat[n_item:]))
            self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, sp_i_vec)))

        self.p += 1

    def update(self, e, is_batch_train=False):
        # static baseline; w/o updating the model
        if not is_batch_train and self.is_static:
            return

        # create user/item ID vector
        x = np.zeros(self.n_user + self.n_item)
        x[e.user.index] = x[self.n_user + e.item.index] = 1.

        # append contextual variables
        x = np.concatenate((x,
                            e.user.feature,
                            e.context,
                            e.item.feature))

        self.update_params(x, e.value)

    def score(self, user, candidates, context):
        # i_mat is (n_item_context, n_item) for all possible items
        # extract only target items
        i_mat = self.i_mat[:, candidates]

        n_target = len(candidates)

        # u_mat will be (n_user + n_user_context, n_item) for the target user
        u_vec = np.concatenate((np.zeros(self.n_user),
                                user.feature,
                                context))
        u_vec[user.index] = 1.
        u_vec = np.array([u_vec]).T
        u_mat = sp.csr_matrix(np.repeat(u_vec, n_target, axis=1))

        # stack them into (p, n_item) matrix
        # rows are ordered by [user ID - item ID - user context - others - item context]
        mat = sp.vstack((u_mat[:self.n_user],
                         i_mat[:self.n_item],
                         u_mat[self.n_user:],
                         i_mat[self.n_item:]))

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

        return np.abs(1. - np.ravel(pred))

    def recommend(self, user, candidates, context):
        scores = self.score(user, candidates, context)
        return self.scores2recos(scores, candidates)
