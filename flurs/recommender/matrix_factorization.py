from ..base import RecommenderMixin
from ..model import MatrixFactorization
from .. import logger

import numpy as np


class MFRecommender(MatrixFactorization, RecommenderMixin):

    __doc__ = MatrixFactorization.__doc__

    def initialize(self, static=False):
        """Initialize a recommender by resetting stored users and items.
        Default the number of users and items to zero.

        Parameters
        ----------
        static : bool, default=False
            Disable incremental update if ``True``s.
        """
        super(MFRecommender, self).initialize()

        # if True, parameters will not be updated in evaluation
        self.static = static

    def register_user(self, user):
        super(MFRecommender, self).register_user(user)
        self.users[user.index]["vec"] = np.random.normal(0.0, 0.1, self.k)

    def register_item(self, item):
        super(MFRecommender, self).register_item(item)
        i_vec = np.random.normal(0.0, 0.1, (1, self.k))
        if self.Q.size == 0:
            self.Q = i_vec
        else:
            self.Q = np.concatenate((self.Q, i_vec))

    def update(self, e, batch_train=False):
        # static baseline; w/o updating the model
        if not batch_train and self.static:
            return

        if e.value != 1.0:
            logger.info(
                "Incremental matrix factorization assumes implicit feedback"
                " recommendation, so the event value is automatically converted"
                " into 1.0"
            )
            e.value = 1.0

        self.update_model(e.user.index, e.item.index, e.value)

    def score(self, user, candidates):
        pred = np.dot(self.users[user.index]["vec"], self.Q[candidates, :].T)
        return np.abs(1.0 - pred.flatten())

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates)
