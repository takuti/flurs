from flurs.recommender import recommender

import numpy as np


class Random(recommender.Recommender):

    """Random baseline
    """

    def __init__(self):
        super().__init__()

    def init_model(self):
        pass

    def add_user(self, u):
        super().add_user(u)

    def add_item(self, i):
        super().add_item(i)

    def update(self, u, i, r, is_batch_train=False):
        pass

    def recommend(self, u, target_i_indices):
        scores = np.random.rand(len(target_i_indices))
        return self.scores2recos(scores, target_i_indices)
