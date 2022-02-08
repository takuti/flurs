from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from flurs.types import User, Item


class UserTestCase(TestCase):
    def test_user(self):
        user = User(1, np.arange(5))
        self.assertEqual(user.index, 1)

        v = user.encode(dim=None, index=True, feature=True, vertical=False)
        assert_array_equal(v, np.array([0, 1, 0, 1, 2, 3, 4]))

        v = user.encode(dim=3, index=True, feature=False, vertical=False)
        assert_array_equal(v, np.array([0, 1, 0]))

        v = user.encode(dim=None, index=False, feature=True, vertical=True)
        assert_array_equal(v, np.array([[0], [1], [2], [3], [4]]))


class ItemTestCase(TestCase):
    def test_item(self):
        item = Item(1, np.arange(5))
        self.assertEqual(item.index, 1)

        v = item.encode(dim=None, index=True, feature=True, vertical=False)
        assert_array_equal(v, np.array([0, 1, 0, 1, 2, 3, 4]))

        v = item.encode(dim=3, index=True, feature=False, vertical=False)
        assert_array_equal(v, np.array([0, 1, 0]))

        v = item.encode(dim=None, index=False, feature=True, vertical=True)
        assert_array_equal(v, np.array([[0], [1], [2], [3], [4]]))
