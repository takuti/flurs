from flurs.base import RecommenderMixin
from flurs.model.bprmf import BPRMF

import numpy as np


class BPRMFRecommender(BPRMF, RecommenderMixin):

    def init_recommender(self):
        super().init_recommender()

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
        self.update_params(e.user.index, e.item.index)

    def score(self, user, candidates):
        pred = np.dot(self.users[user.index]['vec'],
                      self.Q[candidates, :].T)
        return pred.flatten()

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates, rev=True)
