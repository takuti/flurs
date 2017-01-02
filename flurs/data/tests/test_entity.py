from unittest import TestCase
import numpy as np

from flurs.data.entity import User, Item, Event


class EntityTestCase(TestCase):

    def test_user(self):
        user = User(1, np.arange(10))
        self.assertEqual(user.index, 1)

    def test_item(self):
        item = Item(1, np.arange(10))
        self.assertEqual(item.index, 1)

    def test_event(self):
        user = User(1, np.arange(10))
        item = Item(1, np.arange(10))
        event = Event(user, item, 5.0, np.arange(10))
        self.assertEqual(event.value, 5.0)
