from ..base import FeatureRecommenderMixin
from ..model import OnlineSketch

import numpy as np
import numpy.linalg as ln
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.utils.extmath import safe_sparse_dot


class SketchRecommender(OnlineSketch, FeatureRecommenderMixin):

    """Online-matrix-sketching-based recommender

    References
    ----------

    - T. Kitazawa.
      `Sketching Dynamic User-Item Interactions for Online Item Recommendation <http://dl.acm.org/citation.cfm?id=3022152>`_.
      In *Proc. of CHIIR 2017*, March 2017.
    """

    def initialize(self):
        super(SketchRecommender, self).initialize()

    def register_user(self, user):
        super(SketchRecommender, self).register_user(user)

    def register_item(self, item):
        super(SketchRecommender, self).register_item(item)

        i_vec = item.encode(index=False, feature=True, vertical=True)
        i_vec = sp.csr_matrix(i_vec)
        if self.i_mat.size == 0:
            self.i_mat = i_vec
        else:
            self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, i_vec)))

    def update(self, e, batch_train=False):
        y = e.encode(index=False, feature=True, context=True)
        self.update_model(y)

    def score(self, user, candidates, context):
        # i_mat is (n_item_context, n_item) for all possible items
        # extract only target items
        i_mat = self.i_mat[:, candidates]

        n_target = len(candidates)

        # u_mat will be (n_user_context, n_item) for the target user
        u_vec = np.concatenate((user.feature, context))
        u_vec = np.array([u_vec]).T

        u_mat = sp.csr_matrix(np.repeat(u_vec, n_target, axis=1))

        # stack them into (p, n_item) matrix
        Y = sp.vstack((u_mat, i_mat))
        Y = self.proj.reduce(Y)
        Y = sp.csr_matrix(preprocessing.normalize(Y, norm='l2', axis=0))

        X = np.identity(self.k) - np.dot(self.U_r, self.U_r.T)
        A = safe_sparse_dot(X, Y, dense_output=True)

        return ln.norm(A, axis=0, ord=2)

    def recommend(self, user, candidates, context):
        scores = self.score(user, candidates, context)
        return self.scores2recos(scores, candidates)
