from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from flurs.types import User, Item, Event


class EventTestCase(TestCase):
    def test_event(self):
        user = User(1, np.arange(3))
        item = Item(1, np.arange(3))
        event = Event(user, item, 5.0, np.arange(5))
        self.assertEqual(event.value, 5.0)

        v = event.encode(index=False, feature=True, context=True, vertical=False)
        assert_array_equal(v, np.array([0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2]))

        v = event.encode(index=True, feature=True, context=False, vertical=False)
        assert_array_equal(v, np.array([0, 1, 0, 1, 2, 0, 1, 0, 1, 2]))

        v = event.encode(
            n_user=3, n_item=3, index=True, feature=False, context=False, vertical=True
        )
        assert_array_equal(v, np.array([[0], [1], [0], [0], [1], [0]]))
