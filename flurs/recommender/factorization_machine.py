from ..base import FeatureRecommenderMixin
from ..model import FactorizationMachine
from .. import logger

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


class FMRecommender(FactorizationMachine, FeatureRecommenderMixin):

    """Incremental Factorization Machines (FMs) recommender

    References
    ----------

    - T. Kitazawa.
      `Incremental Factorization Machines for Persistently Cold-Starting Online Item Recommendation <https://arxiv.org/abs/1607.02858>`_.
      arXiv:1607.02858 [cs.LG], July 2016.
    """

    def initialize(self, static=False, use_index=False):
        super(FMRecommender, self).initialize()
        self.static = static
        self.use_index = use_index

    def register_user(self, user):
        super(FMRecommender, self).register_user(user)

        if self.use_index:
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

    def register_item(self, item):
        super(FMRecommender, self).register_item(item)

        n_item = self.n_item - 1

        # update the item matrix for all items
        i_vec = item.encode(dim=(n_item + 1),
                            index=self.use_index,
                            feature=True,
                            vertical=True)
        sp_i_vec = sp.csr_matrix(i_vec)

        if self.i_mat.size == 0:
            self.i_mat = sp_i_vec
        elif self.use_index:
            self.i_mat = sp.vstack((self.i_mat[:n_item],
                                    np.zeros((1, n_item)),
                                    self.i_mat[n_item:]))
            self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, sp_i_vec)))
        else:
            self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, sp_i_vec)))

        if self.use_index:

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
            self.p += 1

    def update(self, e, batch_train=False):
        # static baseline; w/o updating the model
        if not batch_train and self.static:
            return

        x = e.encode(index=self.use_index,
                     n_user=self.n_user, n_item=self.n_item)

        if e.value != 1.:
            logger.info('Incremental factorization machines assumes implicit feedback recommendation, so the event value is automatically converted into 1.0')
            e.value = 1.

        self.update_model(x, e.value)

    def score(self, user, candidates, context):
        # i_mat is (n_item_context, n_item) for all possible items
        # extract only target items
        i_mat = self.i_mat[:, candidates]

        n_target = len(candidates)

        u_vec = user.encode(dim=self.n_user,
                            index=self.use_index,
                            feature=True,
                            vertical=True)
        u_vec = np.concatenate((u_vec, np.array([context]).T))
        u_mat = sp.csr_matrix(np.repeat(u_vec, n_target, axis=1))

        mat = sp.vstack((u_mat, i_mat))

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
