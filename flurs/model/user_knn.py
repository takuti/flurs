from flurs.base import Recommender

import numpy as np


class UserKNN(Recommender):

    """Incremental User-based Collaborative Filtering
    """

    def __init__(self, k=5):
        super().__init__()

        # number of nearest neighbors
        self.k = 5

        self.init_model()

    def init_model(self):
        # user-item matrix
        self.R = np.array([])

        # user-user similarity matrix
        self.S = np.array([])

        # user-user similarity: S = B / (sqrt(C) * sqrt(D))
        self.B = np.array([])
        self.C = np.array([])
        self.D = np.array([])

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
        ua, ia = e.user.index, e.item.index

        prev_r = self.R[ua, ia]
        is_new_submit = (prev_r == 0)
        self.R[ua, ia] = e.value

        prev_mean = self.users[ua]['mean']

        if is_new_submit:
            self.users[ua]['count'] += 1
            self.users[ua]['mean'] = self.R[ua, ia] / self.users[ua]['count'] + (
                self.users[ua]['count'] - 1) / self.users[ua]['count'] * prev_mean
        else:
            self.users[ua]['mean'] = (self.R[ua, ia] - prev_r) / (self.users[ua]['count'] - 1) + prev_mean

        d = self.users[ua]['mean'] - prev_mean

        for uy in range(self.n_user):
            # skip myself
            if uy == ua:
                continue

            e = f = g = 0.

            had_uy_rated_ia = (self.R[uy, ia] != 0)
            if had_uy_rated_ia:
                ua_normalized = self.R[ua, ia] - self.users[ua]['mean']
                uy_normalized = self.R[uy, ia] - self.users[uy]['mean']

                if is_new_submit:
                    e = ua_normalized * uy_normalized
                    f = ua_normalized ** 2
                    g = uy_normalized ** 2
                else:
                    e = (self.R[ua, ia] - prev_r) * uy_normalized
                    f = (self.R[ua, ia] - prev_r) ** 2 + 2 * (self.R[ua, ia] - prev_r) * ua_normalized
                    g = 0.

            for ih in range(self.n_item):
                # only for co-rated items
                if self.R[ua, ih] != 0 and self.R[uy, ih] != 0:
                    e = e - d * (self.R[uy, ih] - self.users[uy]['mean'])
                    f = f + d ** 2 - 2 * d * (self.R[ua, ih] - prev_mean)

            self.B[ua, uy] += e
            self.C[ua, uy] += f
            self.D[ua, uy] += g

        self.S[ua, :] = self.B[ua, :] / (np.sqrt(self.C[ua, :]) * np.sqrt(self.D[ua, :]))

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
            pred[pi] += (numer / denom)

        # Larger pred is more promising,
        # but `scores2recos` sorts in an ascending order.
        return -np.abs(pred)

    def recommend(self, user, candidates):
        scores = self.score(user, candidates)
        return self.scores2recos(scores, candidates)
