from orec.recommender.recommender import Recommender

import numpy as np


class IncrementalMF(Recommender):

    """Incremental Matrix Factorization
    """

    def __init__(self, is_static=False, k=40, l2_reg=.01, learn_rate=.003):
        # if True, parameters will not be updated in evaluation
        self.is_static = is_static

        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self.clear()

    def clear(self):
        self.n_user = 0
        self.users = {}

        self.n_item = 0
        self.items = {}

        self.Q = np.array([])

    def check(self, u, i, u_feature, i_feature):
        is_new_user = u not in self.users
        if is_new_user:
            self.users[u] = {'vec': np.random.normal(0., 0.1, self.k), 'observed': set()}
            self.n_user += 1

        is_new_item = i not in self.items
        if is_new_item:
            self.items[i] = {}
            self.n_item += 1
            i = np.random.normal(0., 0.1, (1, self.k))
            self.Q = i if self.Q.size == 0 else np.concatenate((self.Q, i))

        return is_new_user, is_new_item

    def update(self, u, i, r, context=np.array([]), is_batch_train=False):
        # static baseline; w/o updating the model
        if not is_batch_train and self.is_static:
            return

        u_vec = self.users[u]['vec']
        i_vec = self.Q[i]

        err = r - np.inner(u_vec, i_vec)

        grad = -2. * (err * i_vec - self.l2_reg_u * u_vec)
        next_u_vec = u_vec - self.learn_rate * grad

        grad = -2. * (err * u_vec - self.l2_reg_i * i_vec)
        next_i_vec = i_vec - self.learn_rate * grad

        self.users[u]['vec'] = next_u_vec
        self.Q[i] = next_i_vec

    def recommend(self, u, target_i_indices, context):
        pred = np.dot(self.users[u]['vec'], self.Q[target_i_indices, :].T)
        scores = np.abs(1. - pred.flatten())

        return self.scores2recos(scores, target_i_indices)
