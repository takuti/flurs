from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from flurs.data.entity import User, Item, Event
from flurs.baseline import Popular
from flurs.evaluation import Evaluator


class EvaluatorTestCase(TestCase):
    def setUp(self):
        recommender = Popular()
        recommender.initialize()
        self.evaluator = Evaluator(recommender=recommender, repeat=False)

        self.samples = [
            Event(User(0), Item(0), 1),
            Event(User(0), Item(1), 1),
            Event(User(1), Item(2), 1),
            Event(User(0), Item(3), 1),
            Event(User(2), Item(4), 1),
            Event(User(1), Item(4), 1),
            Event(User(0), Item(5), 1),
            Event(User(2), Item(1), 1),
            Event(User(0), Item(6), 1),
            Event(User(2), Item(0), 1),
        ]

    def test_fit(self):
        self.evaluator.fit(self.samples[:2], self.samples[2:3], n_epoch=1)

        self.assertEqual(self.evaluator.rec.n_user, 2)
        self.assertEqual(self.evaluator.rec.n_item, 3)
        assert_array_equal(self.evaluator.rec.freq, np.array([1, 1, 1]))

    def test_evaluate(self):
        self.evaluator.fit(self.samples[:2], self.samples[2:3], n_epoch=1)

        # convert to list for testing w/ `yield`
        list(self.evaluator.evaluate(self.samples[3:]))

        self.assertEqual(self.evaluator.rec.n_user, 3)
        self.assertEqual(self.evaluator.rec.n_item, 7)
        assert_array_equal(self.evaluator.rec.freq, np.array([2, 2, 1, 1, 2, 1, 1]))
