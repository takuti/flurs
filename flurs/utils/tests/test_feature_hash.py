from unittest import TestCase
import numpy as np

from flurs.utils.feature_hash import n_feature_hash, feature_hash, multiple_feature_hash


class FeatureHashTestCase(TestCase):
    def test_feature_hash(self):
        x = feature_hash("Tom", 5, seed=123)
        nnz = np.nonzero(x)[0].size
        self.assertEqual(nnz, 1)
        self.assertEqual(x.size, 5)

    def test_multiple_feature_hash(self):
        x = multiple_feature_hash("Tom", 5, seed=123)
        nnz = np.nonzero(x)[0].size
        self.assertEqual(nnz, 1)
        self.assertEqual(x.size, 5)

    def test_n_feature_hash(self):
        x = n_feature_hash("Tom", [5, 5, 5], seeds=[123, 456, 789])
        nnz = np.nonzero(x)[0].size
        self.assertEqual(nnz, 3)
        self.assertEqual(x.size, 15)
