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

    def recommend(self, user, candidates):
        scores = np.random.rand(len(candidates))
        return self.scores2recos(scores, candidates)
