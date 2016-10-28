from orec.recommender import Recommender

import numpy as np


class Random(Recommender):

    """Random baseline
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.n_user = 0
        self.users = {}

        self.n_item = 0
        self.items = {}

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

        return is_new_user, is_new_item

    def update(self, d, is_batch_train=False):
        return

    def recommend(self, d, target_i_indices):
        scores = np.random.rand(len(target_i_indices))
        return self.scores2recos(scores, target_i_indices)
