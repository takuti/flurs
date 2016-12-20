from flurs.base import Recommender

import numpy as np


class Popular(Recommender):

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

    def score(self, user, candidates):
        return self.freq[candidates]

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates, rev=True)
