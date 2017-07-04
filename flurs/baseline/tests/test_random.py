from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from flurs.data.entity import User, Item, Event
from flurs.baseline.random import Random


class RandomTestCase(TestCase):

    def setUp(self):
        self.recommender = Random()
        self.recommender.init_recommender()

    def test_add_user(self):
        self.recommender.add_user(User(0))
        self.assertEqual(self.recommender.n_user, 1)

    def test_add_item(self):
        self.recommender.add_item(Item(0))
        self.assertEqual(self.recommender.n_item, 1)

    def test_score(self):
        self.recommender.add_user(User(0))
        self.recommender.add_item(Item(0))
        self.recommender.update(Event(User(0), Item(0), 1))
        score = self.recommender.score(User(0), np.array([0]))
        self.assertTrue(score >= 0. and score < 1.)
