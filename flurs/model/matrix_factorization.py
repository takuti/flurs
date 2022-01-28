from sklearn.base import BaseEstimator

import numpy as np


class MatrixFactorization(BaseEstimator):

    r"""Incremental Matrix Factorization (MF).

    Parameters
    ----------
    k : int, default=40
        Number of latent factors.

    l2_reg : float, default=0.01
        :math:`\lambda` for L2 regularization.

    learn_rate : float, default=0.003
        Learning rate :math:`\eta`.

    References
    ----------
    .. [1] J. Vinagre et al.
           `Fast Incremental Matrix Factorization for Recommendation with Positive-only
           Feedback <http://link.springer.com/chapter/10.1007/978-3-319-08786-3_41>`_.
           In *Proc. of UMAP 2014*, pp. 459-470, July 2014.
    """

    def __init__(self, k=40, l2_reg=0.01, learn_rate=0.003):
        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self.Q = np.array([])

    def update_model(self, ua, ia, value):
        u_vec = self.users[ua]["vec"]
        i_vec = self.Q[ia]

        err = value - np.inner(u_vec, i_vec)

        grad = -2.0 * (err * i_vec - self.l2_reg_u * u_vec)
        next_u_vec = u_vec - self.learn_rate * grad

        grad = -2.0 * (err * u_vec - self.l2_reg_i * i_vec)
        next_i_vec = i_vec - self.learn_rate * grad

        self.users[ua]["vec"] = next_u_vec
        self.Q[ia] = next_i_vec
