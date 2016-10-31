from orec.recommender import recommender

import numpy as np


class Popular(recommender.Recommender):

    """Popularity (non-personalized) baseline
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.n_user = 0
        self.users = {}

        self.n_item = 0
        self.items = {}

        self.freq = np.array([])

    def add_user(self, u):
        super().add_user(u)

    def add_item(self, i):
        super().add_item()
        self.freq = np.append(self.freq, 0)

    def update(self, u, i, r, is_batch_train=False):
        self.freq[i] += 1

    def recommend(self, u, target_i_indices):
        sorted_indices = np.argsort(self.freq[target_i_indices])[::-1]
        return target_i_indices[sorted_indices], self.freq[target_i_indices][sorted_indices]
