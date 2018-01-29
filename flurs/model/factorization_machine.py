from sklearn.base import BaseEstimator

import numpy as np
import scipy.sparse as sp

from .. import logger


class FactorizationMachine(BaseEstimator):

    """Incremental Factorization Machines

    T. Kitazawa.
    "Incremental Factorization Machines for Persistently Cold-Starting Online Item Recommendation"
    arXiv:1607.02858 [cs.LG], July 2016.

    """

    def __init__(self,
                 p=None,
                 k=40,
                 l2_reg_w0=2.,
                 l2_reg_w=8.,
                 l2_reg_V=16.,
                 learn_rate=.004):

        assert p is not None

        # number of dimensions of input vectors
        self.p = p

        self.k = k
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = np.ones(k) * l2_reg_V
        self.learn_rate = learn_rate

        self.i_mat = sp.csr_matrix([])

        # initial parameters
        self.w0 = 0.
        self.w = np.zeros(self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))

        # to keep the last parameters for adaptive regularization
        self.prev_w0 = self.w0
        self.prev_w = self.w.copy()
        self.prev_V = self.V.copy()

    def update_reg(self, x, err):
        x_vec = np.array([x]).T  # p x 1

        # update regularization parameters
        coeff = 4. * self.learn_rate * err * self.learn_rate

        self.l2_reg_w0 = max(0., self.l2_reg_w0 + coeff * self.prev_w0)
        self.l2_reg_w = max(0., self.l2_reg_w + coeff * np.inner(x, self.prev_w))

        if self.l2_reg_w0 == 0. or self.l2_reg_w == 0.:
            logger.warn('reg_w0 and/or reg_w are fallen in 0.0')

        dot_v = np.dot(x_vec.T, self.V).reshape((self.k,))  # (k, )
        dot_prev_v = np.dot(x_vec.T, self.prev_V).reshape((self.k,))  # (k, )
        s_duplicated = np.dot((x_vec.T ** 2), self.V * self.prev_V).reshape((self.k,))  # (k, )
        self.l2_rev_V = np.maximum(np.zeros(self.k), self.l2_reg_V + coeff * (dot_v * dot_prev_v - s_duplicated))

    def update_model(self, x, value):
        x_vec = np.array([x]).T  # p x 1
        interaction = np.sum(np.dot(self.V.T, x_vec) ** 2 - np.dot(self.V.T ** 2, x_vec ** 2)) / 2.
        pred = self.w0 + np.inner(self.w, x) + interaction

        # compute current error
        err = value - pred

        self.update_reg(x, err)

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
