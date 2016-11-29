from flurs.recommender import recommender

import numpy as np


class Popular(recommender.Recommender):

    """Popularity (non-personalized) baseline
    """

    def __init__(self):
        super().__init__()

    def init_model(self):
        self.freq = np.array([])

    def add_user(self, u):
        super().add_user(u)

    def add_item(self, i):
        super().add_item(i)
        self.freq = np.append(self.freq, 0)

    def update(self, u, i, r, is_batch_train=False):
        self.freq[i] += 1

    def recommend(self, u, target_i_indices):
        sorted_indices = np.argsort(self.freq[target_i_indices])[::-1]
        return target_i_indices[sorted_indices], self.freq[target_i_indices][sorted_indices]
