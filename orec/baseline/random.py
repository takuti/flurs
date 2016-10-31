from orec.recommender.recommender import Recommender

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

    def add_user(self, u):
        super().add_user(u)

    def add_item(self, i):
        super().add_item(i)

    def update(self, u, i, r, is_batch_train=False):
        return

    def recommend(self, u, target_i_indices):
        scores = np.random.rand(len(target_i_indices))
        return self.scores2recos(scores, target_i_indices)
