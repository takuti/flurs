import numpy as np


class User:

    def __init__(self, index, feature=np.array([0.])):
        self.index = index
        self.feature = feature

    def __str__(self):
        return 'user %d: %s' % (self.index, self.feature)


class Item:

    def __init__(self, index, feature=np.array([0.])):
        self.index = index
        self.feature = feature

    def __str__(self):
        return 'item %d: %s' % (self.index, self.feature)


class Event:

    def __init__(self, user, item, value, context=np.array([0.])):
        self.user = user
        self.item = item
        self.value = value
        self.context = context
