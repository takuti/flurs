from sklearn.base import BaseEstimator

import numpy as np


class BPRMF(BaseEstimator):

    """Incremental Matrix Factorization with BPR optimization

    S. Rendle et al.
    "BPR: Bayesian Personalized Ranking from Implicit Feedback"
    In Proceedings of UAI 2009, pages 452-461, June 2009.

    """

    def __init__(self, k=40, l2_reg=.01, learn_rate=.003):
        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg  # positive items: i
        self.l2_reg_j = l2_reg  # negative items: j
        self.learn_rate = learn_rate

        self.Q = np.array([])

    def update_model(self, ua, ia):

        u_vec = self.users[ua]['vec']
        i_vec = self.Q[ia]
        x_ui = np.inner(u_vec, i_vec)

        unobserved = list(set(range(self.n_item)) - self.users[ua]['known_items'])

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

        self.users[ua]['vec'] = next_u_vec
        self.Q[ia] = next_i_vec
        self.Q[j] = next_j_vec
