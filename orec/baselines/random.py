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

    def check(self, u, i, u_feature, i_feature):
        is_new_user = u not in self.users
        if is_new_user:
            self.users[u] = {'observed': set()}
            self.n_user += 1

        is_new_item = i not in self.items
        if is_new_item:
            self.items[i] = {}
            self.n_item += 1

        return is_new_user, is_new_item

    def update(self, u, i, r, context=np.array([]), is_batch_train=False):
        return

    def recommend(self, u, target_i_indices, context=np.array([])):
        scores = np.random.rand(len(target_i_indices))
        return self.scores2recos(scores, target_i_indices)
