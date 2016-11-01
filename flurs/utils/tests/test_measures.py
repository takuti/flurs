from unittest import TestCase
import numpy as np

from flurs.utils import measures


class MeasuresTestCase(TestCase):

    def setUp(self):
        self.truth = np.array([1, 2, 4])
        self.recommend = np.array([1, 3, 2, 6, 4, 5])
        self.k = 2

    def test_recall(self):
        actual = measures.recall(self.truth, self.recommend, self.k)
        self.assertAlmostEqual(actual, 0.333, delta=1e-3)

    def test_precision(self):
        actual = measures.precision(self.truth, self.recommend, self.k)
        self.assertAlmostEqual(actual, 0.5, delta=1e-3)

    def test_mean_average_precision(self):
        actual = measures.mean_average_precision(self.truth, self.recommend)
        self.assertAlmostEqual(actual, 0.756, delta=1e-3)

    def test_auc(self):
        actual = measures.auc(self.truth, self.recommend)
        self.assertAlmostEqual(actual, 0.667, delta=1e-3)

    def test_mrr(self):
        actual = measures.mrr(self.truth, self.recommend)
        self.assertAlmostEqual(actual, 1.0, delta=1e-3)

    def test_mpr(self):
        actual = measures.mpr(self.truth, self.recommend)
        self.assertAlmostEqual(actual, 33.333, delta=1e-3)

    def test_ndcg(self):
        actual = measures.ndcg(self.truth, self.recommend, self.k)
        self.assertAlmostEqual(actual, 0.613, delta=1e-3)
