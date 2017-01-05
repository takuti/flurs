from flurs.base import RecommenderMixin
from flurs.model.mf import MatrixFactorization

import numpy as np


class MFRecommender(MatrixFactorization, RecommenderMixin):

    def init_recommender(self, is_static=False):
        super().init_recommender()

        # if True, parameters will not be updated in evaluation
        self.is_static = is_static

    def add_user(self, user):
        super().add_user(user)
        self.users[user.index]['vec'] = np.random.normal(0., 0.1, self.k)

    def add_item(self, item):
        super().add_item(item)
        i_vec = np.random.normal(0., 0.1, (1, self.k))
        if self.Q.size == 0:
            self.Q = i_vec
        else:
            self.Q = np.concatenate((self.Q, i_vec))

    def update(self, e, is_batch_train=False):
        # static baseline; w/o updating the model
        if not is_batch_train and self.is_static:
            return

        self.update_params(e.user.index, e.item.index, e.value)

    def score(self, user, candidates):
        pred = np.dot(self.users[user.index]['vec'],
                      self.Q[candidates, :].T)
        return np.abs(1. - pred.flatten())

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates)
