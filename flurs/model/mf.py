from sklearn.base import BaseEstimator

import numpy as np


class MatrixFactorization(BaseEstimator):

    """Incremental Matrix Factorization

    J. Vinagre et al.
    "Fast Incremental Matrix Factorization for Recommendation with Positive-Only Feedback"
    In Proceedings of UMAP 2014, pages 459-470, July 2014.

    """

    def __init__(self, k=40, l2_reg=.01, learn_rate=.003):
        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self.Q = np.array([])

    def update_model(self, ua, ia, value):
        u_vec = self.users[ua]['vec']
        i_vec = self.Q[ia]

        err = value - np.inner(u_vec, i_vec)

        grad = -2. * (err * i_vec - self.l2_reg_u * u_vec)
        next_u_vec = u_vec - self.learn_rate * grad

        grad = -2. * (err * u_vec - self.l2_reg_i * i_vec)
        next_i_vec = i_vec - self.learn_rate * grad

        self.users[ua]['vec'] = next_u_vec
        self.Q[ia] = next_i_vec
