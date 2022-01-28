from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from flurs.data.entity import User, Item, Event
from flurs.baseline import Popular


class PopularTestCase(TestCase):
    def setUp(self):
        self.recommender = Popular()
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
        self.recommender.update(Event(User(0), Item(0), 1))
        self.assertEqual(self.recommender.n_user, 1)
        self.assertEqual(self.recommender.n_item, 1)
        assert_array_equal(self.recommender.freq, np.array([1]))

    def test_score(self):
        self.recommender.register(User(0))
        self.recommender.register(Item(0))
        self.recommender.update(Event(User(0), Item(0), 1))
        self.assertEqual(self.recommender.score(User(0), np.array([0])), 1)
