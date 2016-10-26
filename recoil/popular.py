from .base import Base

import numpy as np


class Popular(Base):

    """Popularity (non-personalized) baseline
    """

    def __init__(self):
        self._Base__clear()

    def _Base__clear(self):
        self.n_user = 0
        self.users = {}

        self.n_item = 0
        self.items = {}

        self.freq = np.array([])

    def _Base__check(self, d):
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

    def _Base__update(self, d, is_batch_train=False):
        self.freq[d['i_index']] += 1

    def _Base__recommend(self, d, target_i_indices):
        sorted_indices = np.argsort(self.freq[target_i_indices])[::-1]
        return target_i_indices[sorted_indices], self.freq[target_i_indices][sorted_indices]
