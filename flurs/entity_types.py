import numpy as np


class User:

    def __init__(self, index, feature=np.array([0.])):
        self.index = index
        self.feature = feature


class Item:

    def __init__(self, index, feature=np.array([0.])):
        self.index = index
        self.feature = feature


class Event:

    def __init__(self, user, item, value, context=np.array([0.])):
        self.user = user
        self.item = item
        self.value = value
        self.context = context
