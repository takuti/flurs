from sklearn.base import BaseEstimator

import numpy as np


class UserKNN(BaseEstimator):

    """Incremental User-based Collaborative Filtering

    M. Pepagelis et al.
    "Incremental Collaborative Filtering for Highly-Scalable Recommendation Algorithms"
    In Foundations of Intelligent Systems, pages 553-561. Springer Berlin Heidelberg, 2005.

    """

    def __init__(self, k=5):
        # number of nearest neighbors
        self.k = k

        # user-item matrix
        self.R = np.array([])

        # user-user similarity matrix
        self.S = np.array([])

        # user-user similarity: S = B / (sqrt(C) * sqrt(D))
        self.B = np.array([])
        self.C = np.array([])
        self.D = np.array([])

    def update_model(self, ua, ia, value):
        prev_r = self.R[ua, ia]
        new_submit = (prev_r == 0)
        self.R[ua, ia] = value

        prev_mean = self.users[ua]['mean']

        if new_submit:
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

                if new_submit:
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

        # avoid zero division
        idx = (self.C[ua, :] != 0) & (self.C[ua, :] != 0)
        self.S[ua, idx] = self.B[ua, idx] / (np.sqrt(self.C[ua, idx]) * np.sqrt(self.D[ua, idx]))
