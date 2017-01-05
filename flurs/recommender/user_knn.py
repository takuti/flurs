from flurs.base import RecommenderMixin
from flurs.model.user_knn import UserKNN

import numpy as np


class UserKNNRecommender(UserKNN, RecommenderMixin):

    def init_recommender(self):
        super().init_recommender()

    def insert_row(self, mat, n_col):
        row = np.zeros((1, n_col))
        if mat.size == 0:
            mat = row
        else:
            mat = np.concatenate((mat, row))
        return mat

    def insert_col(self, mat, n_row):
        col = np.zeros((n_row, 1))
        if mat.size == 0:
            mat = col
        else:
            mat = np.concatenate((mat, col), axis=1)
        return mat

    def add_user(self, user):
        super().add_user(user)

        self.R = self.insert_row(self.R, self.n_item)

        self.S = self.insert_row(self.S, self.n_user - 1)
        self.S = self.insert_col(self.S, self.n_user)

        self.B = self.insert_row(self.B, self.n_user - 1)
        self.B = self.insert_col(self.B, self.n_user)

        self.C = self.insert_row(self.C, self.n_user - 1)
        self.C = self.insert_col(self.C, self.n_user)

        self.D = self.insert_row(self.D, self.n_user - 1)
        self.D = self.insert_col(self.D, self.n_user)

        # keep track how many times each user interacted with items
        # to compute user's mean
        self.users[user.index]['count'] = 0

        # current average rating of the user
        self.users[user.index]['mean'] = 0.

    def add_item(self, item):
        super().add_item(item)
        self.R = self.insert_col(self.R, self.n_user)

    def update(self, e, is_batch_train=False):
        self.update_params(e.user.index, e.item.index, e.value)

    def score(self, user, candidates):
        ua = user.index

        # find k nearest neightbors
        top_uys = np.argsort(self.S[ua, :])[::-1][:self.k]

        pred = np.ones(len(candidates)) * self.users[ua]['mean']
        for pi, ii in enumerate(candidates):
            denom = numer = 0.
            for uy in top_uys:
                numer += ((self.R[uy, ii] - self.users[uy]['mean']) * self.S[ua, uy])
                denom += self.S[ua, uy]
            if denom != 0.:
                pred[pi] += (numer / denom)

        return np.abs(pred)

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates, rev=True)
