from flurs.recommender import recommender

import numpy as np


class Popular(recommender.Recommender):

    """Popularity (non-personalized) baseline
    """

    def __init__(self):
        super().__init__()

    def init_model(self):
        self.freq = np.array([])

    def add_user(self, user):
        super().add_user(user)

    def add_item(self, item):
        super().add_item(item)
        self.freq = np.append(self.freq, 0)

    def update(self, e, is_batch_train=False):
        self.freq[e.item.index] += 1

    def recommend(self, user, target_i_indices):
        sorted_indices = np.argsort(self.freq[target_i_indices])[::-1]
        return target_i_indices[sorted_indices], self.freq[target_i_indices][sorted_indices]
