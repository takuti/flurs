from orec.recommender import Recommender

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

    def check(self, d):
        u_index = d['u_index']
        is_new_user = u_index not in self.users
        if is_new_user:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

        i_index = d['i_index']
        is_new_item = i_index not in self.items
        if is_new_item:
            self.items[i_index] = {}
            self.n_item += 1
            self.freq = np.append(self.freq, 0)

        return is_new_user, is_new_item

    def update(self, d, is_batch_train=False):
        self.freq[d['i_index']] += 1

    def recommend(self, d, target_i_indices):
        sorted_indices = np.argsort(self.freq[target_i_indices])[::-1]
        return target_i_indices[sorted_indices], self.freq[target_i_indices][sorted_indices]
