from sklearn.base import BaseEstimator
from flurs.base import RecommenderMixin

import numpy as np


class Popular(BaseEstimator, RecommenderMixin):

    """Popularity (non-personalized) baseline
    """

    def __init__(self):
        self.freq = np.array([])

    def update(self, ia):
        self.freq[ia] += 1

    def init_recommender(self):
        super(Popular, self).init_recommender()

    def register_user(self, user):
        super(Popular, self).register_user(user)

    def register_item(self, item):
        super(Popular, self).register_item(item)
        self.freq = np.append(self.freq, 0)

    def update_recommender(self, e, batch_train=False):
        self.update(e.item.index)

    def score(self, user, candidates):
        return self.freq[candidates]

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates, rev=True)
