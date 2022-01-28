from ..base import RecommenderMixin
from ..model import UserKNN

import numpy as np


class UserKNNRecommender(UserKNN, RecommenderMixin):

    __doc__ = UserKNN.__doc__

    def initialize(self):
        super(UserKNNRecommender, self).initialize()

    def register_user(self, user):
        super(UserKNNRecommender, self).register_user(user)

        self.R = self.__insert_row(self.R, self.n_item)

        self.S = self.__insert_row(self.S, self.n_user - 1)
        self.S = self.__insert_col(self.S, self.n_user)

        self.B = self.__insert_row(self.B, self.n_user - 1)
        self.B = self.__insert_col(self.B, self.n_user)

        self.C = self.__insert_row(self.C, self.n_user - 1)
        self.C = self.__insert_col(self.C, self.n_user)

        self.D = self.__insert_row(self.D, self.n_user - 1)
        self.D = self.__insert_col(self.D, self.n_user)

        # keep track how many times each user interacted with items
        # to compute user's mean
        self.users[user.index]["count"] = 0

        # current average rating of the user
        self.users[user.index]["mean"] = 0.0

    def register_item(self, item):
        super(UserKNNRecommender, self).register_item(item)
        self.R = self.__insert_col(self.R, self.n_user)

    def update(self, e, batch_train=False):
        self.update_model(e.user.index, e.item.index, e.value)

    def score(self, user, candidates):
        ua = user.index

        # find k nearest neightbors
        top_uys = np.argsort(self.S[ua, :])[::-1][: self.k]

        pred = np.ones(len(candidates)) * self.users[ua]["mean"]
        for pi, ii in enumerate(candidates):
            denom = numer = 0.0
            for uy in top_uys:
                numer += (self.R[uy, ii] - self.users[uy]["mean"]) * self.S[ua, uy]
                denom += self.S[ua, uy]
            if denom != 0.0:
                pred[pi] += numer / denom

        return np.abs(pred)

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates, rev=True)

    def __insert_row(self, mat, n_col):
        row = np.zeros((1, n_col))
        if mat.size == 0:
            mat = row
        else:
            mat = np.concatenate((mat, row))
        return mat

    def __insert_col(self, mat, n_row):
        col = np.zeros((n_row, 1))
        if mat.size == 0:
            mat = col
        else:
            mat = np.concatenate((mat, col), axis=1)
        return mat
