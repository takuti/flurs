from flurs.base import Recommender

import numpy as np


class BPRMF(Recommender):

    """Incremental Matrix Factorization with BPR optimization
    """

    def __init__(self, k=40, l2_reg=.01, learn_rate=.003):
        super().__init__()

        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg  # positive items: i
        self.l2_reg_j = l2_reg  # negative items: j
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

        u_vec = self.users[e.user.index]['vec']
        i_vec = self.Q[e.item.index]
        x_ui = np.inner(u_vec, i_vec)

        unobserved = list(set(range(self.n_item)) - self.users[e.user.index]['observed'])

        # choose one negative (i.e., unobserved) sample
        j = np.random.choice(unobserved)

        j_vec = self.Q[j]

        x_uj = np.inner(u_vec, j_vec)
        x_uij = x_ui - x_uj
        sigmoid = np.e ** (-x_uij) / (1 + np.e ** (-x_uij))

        grad = i_vec - j_vec
        next_u_vec = u_vec + self.learn_rate * (sigmoid * grad + self.l2_reg_u * u_vec)

        grad = u_vec
        next_i_vec = i_vec + self.learn_rate * (sigmoid * grad + self.l2_reg_i * i_vec)

        grad = -u_vec
        next_j_vec = j_vec + self.learn_rate * (sigmoid * grad + self.l2_reg_j * j_vec)

        self.users[e.user.index]['vec'] = next_u_vec
        self.Q[e.item.index] = next_i_vec
        self.Q[j] = next_j_vec

    def recommend(self, user, candidates):
        pred = np.dot(self.users[user.index]['vec'],
                      self.Q[candidates, :].T)
        scores = pred.flatten()

        return self.scores2recos(scores, candidates, rev=True)
