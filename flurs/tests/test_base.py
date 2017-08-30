from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from flurs.base import RecommenderMixin


class RecommenderMixinTestCase(TestCase):

    def setUp(self):
        self.recommender = RecommenderMixin()

    def test_initialize(self):
        self.recommender.initialize()
        self.assertEqual(self.recommender.n_user, 0)
        self.assertEqual(self.recommender.users, {})
        self.assertEqual(self.recommender.n_item, 0)
        self.assertEqual(self.recommender.items, {})

    def test_is_new_user(self):
        self.recommender.initialize()

        self.assertTrue(self.recommender.is_new_user(1))
        self.assertTrue(self.recommender.is_new_user(2))

        self.recommender.users[1] = {}
        self.assertFalse(self.recommender.is_new_user(1))

    def test_is_new_item(self):
        self.recommender.initialize()

        self.assertTrue(self.recommender.is_new_item(1))
        self.assertTrue(self.recommender.is_new_item(2))

        self.recommender.items[1] = {}
        self.assertFalse(self.recommender.is_new_item(1))

    def test_scores2recos(self):
        candidates = np.array([10, 100, 1000])
        scores = np.array([1., 5., 3.])

        candidates_, scores_ = self.recommender.scores2recos(scores, candidates)
        assert_array_equal(scores_, np.array([1., 3., 5.]))
        assert_array_equal(candidates_, np.array([10, 1000, 100]))

        candidates_, scores_ = self.recommender.scores2recos(scores, candidates, rev=True)
        assert_array_equal(scores_, np.array([5., 3., 1.]))
        assert_array_equal(candidates_, np.array([100, 1000, 10]))
