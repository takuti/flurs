from unittest import TestCase
import numpy as np

from flurs.types import User, Item, Event
from flurs.recommender import FMRecommender


class FMRecommenderTestCase(TestCase):
    def setUp(self):
        self.k = 40
        self.recommender = FMRecommender(p=3, k=self.k)
        self.recommender.initialize()

    def test_register_user(self):
        self.recommender.register(User(0))
        self.assertEqual(self.recommender.n_user, 1)

    def test_register_item(self):
        self.recommender.register(Item(0))
        self.assertEqual(self.recommender.n_item, 1)

    def test_update(self):
        self.recommender.register(User(0))
        self.recommender.register(Item(0))
        self.recommender.update(Event(User(0), Item(0), 1, context=np.array([1, 2, 3])))
        self.assertEqual(self.recommender.n_user, 1)
        self.assertEqual(self.recommender.n_item, 1)

    def test_score(self):
        self.recommender.register(User(0))
        self.recommender.register(Item(0))
        self.recommender.update(Event(User(0), Item(0), 1, context=np.array([1, 2, 3])))
        score = self.recommender.score(
            User(0), candidates=np.array([0]), context=np.array([1, 2, 3])
        )
        self.assertTrue(score >= 0.0)
