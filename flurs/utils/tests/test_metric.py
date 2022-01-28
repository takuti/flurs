from unittest import TestCase
import numpy as np

from flurs.utils import metric


class MetricTestCase(TestCase):
    def setUp(self):
        self.truth = np.array([1, 2, 4])
        self.recommend = np.array([1, 3, 2, 6, 4, 5])
        self.k = 2

    def test_recall(self):
        actual = metric.recall(self.truth, self.recommend, self.k)
        self.assertAlmostEqual(actual, 0.333, delta=1e-3)

    def test_precision(self):
        actual = metric.precision(self.truth, self.recommend, self.k)
        self.assertAlmostEqual(actual, 0.5, delta=1e-3)

    def test_average_precision(self):
        actual = metric.average_precision(self.truth, self.recommend)
        self.assertAlmostEqual(actual, 0.756, delta=1e-3)

    def test_auc(self):
        actual = metric.auc(self.truth, self.recommend)
        self.assertAlmostEqual(actual, 0.667, delta=1e-3)

    def test_reciprocal_rank(self):
        actual = metric.reciprocal_rank(self.truth, self.recommend)
        self.assertAlmostEqual(actual, 1.0, delta=1e-3)

    def test_mpr(self):
        actual = metric.mpr(self.truth, self.recommend)
        self.assertAlmostEqual(actual, 33.333, delta=1e-3)

    def test_ndcg(self):
        actual = metric.ndcg(self.truth, self.recommend, self.k)
        self.assertAlmostEqual(actual, 0.613, delta=1e-3)
