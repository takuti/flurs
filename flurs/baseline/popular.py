from flurs.base import BaseModel, RecommenderMixin

import numpy as np


class Popular(BaseModel, RecommenderMixin):

    """Popularity (non-personalized) baseline
    """

    def __init__(self):
        pass

    def init_params(self):
        self.freq = np.array([])

    def update_params(self, ia):
        self.freq[ia] += 1

    def init_recommender(self):
        super().init_recommender()

    def add_user(self, user):
        super().add_user(user)

    def add_item(self, item):
        super().add_item(item)
        self.freq = np.append(self.freq, 0)

    def update(self, e, is_batch_train=False):
        self.update_params(e.item.index)

    def score(self, user, candidates):
        return self.freq[candidates]

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates, rev=True)
