from flurs.base import BaseModel, RecommenderMixin

import numpy as np


class Random(BaseModel, RecommenderMixin):

    """Random baseline
    """

    def __init__(self):
        pass

    def init_recommender(self):
        super(Random, self).init_recommender()

    def add_user(self, user):
        super(Random, self).add_user(user)

    def add_item(self, item):
        super(Random, self).add_item(item)

    def update_recommender(self, e, is_batch_train=False):
        pass

    def score(self, user, candidates):
        return np.random.rand(len(candidates))

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates)
