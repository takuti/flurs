from unittest import TestCase
import numpy as np

from flurs.data.entity import User, Item, Event
from flurs.baseline.random import Random


class RandomTestCase(TestCase):

    def setUp(self):
        self.recommender = Random()
        self.recommender.initialize()

    def test_register_user(self):
        self.recommender.register(User(0))
        self.assertEqual(self.recommender.n_user, 1)

    def test_register_item(self):
        self.recommender.register(Item(0))
        self.assertEqual(self.recommender.n_item, 1)

    def test_score(self):
        self.recommender.register(User(0))
        self.recommender.register(Item(0))
        self.recommender.update(Event(User(0), Item(0), 1))
        score = self.recommender.score(User(0), np.array([0]))
        self.assertTrue(score >= 0. and score < 1.)
