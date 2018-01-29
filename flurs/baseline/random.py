from sklearn.base import BaseEstimator
from ..base import RecommenderMixin

import numpy as np


class Random(BaseEstimator, RecommenderMixin):

    """Random baseline
    """

    def __init__(self):
        pass

    def initialize(self):
        super(Random, self).initialize()

    def register_user(self, user):
        super(Random, self).register_user(user)

    def register_item(self, item):
        super(Random, self).register_item(item)

    def update(self, e, batch_train=False):
        pass

    def score(self, user, candidates):
        return np.random.rand(len(candidates))

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates)
