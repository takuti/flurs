from flurs.base import Recommender

import numpy as np


class Random(Recommender):

    """Random baseline
    """

    def __init__(self):
        super().__init__()

    def init_model(self):
        pass

    def add_user(self, user):
        super().add_user(user)

    def add_item(self, item):
        super().add_item(item)

    def update(self, e, is_batch_train=False):
        pass

    def score(self, user, candidates):
        return np.random.rand(len(candidates))

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates)
