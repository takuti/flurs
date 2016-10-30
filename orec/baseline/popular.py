from orec.recommender.recommender import Recommender

import numpy as np


class Popular(Recommender):

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

    def check(self, u, i):
        is_new_user = u not in self.users
        if is_new_user:
            self.users[u] = {'observed': set()}
            self.n_user += 1

        is_new_item = i not in self.items
        if is_new_item:
            self.items[i] = {}
            self.n_item += 1
            self.freq = np.append(self.freq, 0)

        return is_new_user, is_new_item

    def update(self, u, i, r, is_batch_train=False):
        self.freq[i] += 1

    def recommend(self, u, target_i_indices):
        sorted_indices = np.argsort(self.freq[target_i_indices])[::-1]
        return target_i_indices[sorted_indices], self.freq[target_i_indices][sorted_indices]
