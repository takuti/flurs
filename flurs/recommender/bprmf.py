from ..base import RecommenderMixin
from ..model import BPRMF

import numpy as np


class BPRMFRecommender(BPRMF, RecommenderMixin):

    __doc__ = BPRMF.__doc__

    def initialize(self):
        super(BPRMFRecommender, self).initialize()

    def register_user(self, user):
        super(BPRMFRecommender, self).register_user(user)
        self.users[user.index]["vec"] = np.random.normal(0.0, 0.1, self.k)

    def register_item(self, item):
        super(BPRMFRecommender, self).register_item(item)
        i_vec = np.random.normal(0.0, 0.1, (1, self.k))
        if self.Q.size == 0:
            self.Q = i_vec
        else:
            self.Q = np.concatenate((self.Q, i_vec))

    def update(self, e, batch_train=False):
        self.update_model(e.user.index, e.item.index)

    def score(self, user, candidates):
        pred = np.dot(self.users[user.index]["vec"], self.Q[candidates, :].T)
        return pred.flatten()

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates, rev=True)
