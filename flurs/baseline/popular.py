from sklearn.base import BaseEstimator
from ..base import RecommenderMixin

import numpy as np


class Popular(BaseEstimator, RecommenderMixin):

    """Popularity (non-personalized) baseline
    """

    def __init__(self):
        self.freq = np.array([])

    def initialize(self):
        super(Popular, self).initialize()

    def register_user(self, user):
        super(Popular, self).register_user(user)

    def register_item(self, item):
        super(Popular, self).register_item(item)
        self.freq = np.append(self.freq, 0)

    def update(self, e, batch_train=False):
        self.freq[e.item.index] += 1

    def score(self, user, candidates):
        return self.freq[candidates]

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates, rev=True)
