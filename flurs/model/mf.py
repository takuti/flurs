from flurs.base import Recommender

import numpy as np


class MatrixFactorization(Recommender):

    """Incremental Matrix Factorization
    """

    def __init__(self, is_static=False, k=40, l2_reg=.01, learn_rate=.003):
        super().__init__()

        # if True, parameters will not be updated in evaluation
        self.is_static = is_static

        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self.init_model()

    def init_model(self):
        self.Q = np.array([])

    def add_user(self, user):
        super().add_user(user)
        self.users[user.index]['vec'] = np.random.normal(0., 0.1, self.k)

    def add_item(self, item):
        super().add_item(item)
        i_vec = np.random.normal(0., 0.1, (1, self.k))
        if self.Q.size == 0:
            self.Q = i_vec
        else:
            self.Q = np.concatenate((self.Q, i_vec))

    def update(self, e, is_batch_train=False):
        # static baseline; w/o updating the model
        if not is_batch_train and self.is_static:
            return

        u_vec = self.users[e.user.index]['vec']
        i_vec = self.Q[e.item.index]

        err = e.value - np.inner(u_vec, i_vec)

        grad = -2. * (err * i_vec - self.l2_reg_u * u_vec)
        next_u_vec = u_vec - self.learn_rate * grad

        grad = -2. * (err * u_vec - self.l2_reg_i * i_vec)
        next_i_vec = i_vec - self.learn_rate * grad

        self.users[e.user.index]['vec'] = next_u_vec
        self.Q[e.item.index] = next_i_vec

    def recommend(self, user, candidates):
        pred = np.dot(self.users[user.index]['vec'],
                      self.Q[candidates, :].T)
        scores = np.abs(1. - pred.flatten())

        return self.scores2recos(scores, candidates)
