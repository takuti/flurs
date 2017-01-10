import numpy as np


class Base:

    def __init__(self, index, feature=np.array([0.])):
        self.index = index
        self.feature = feature

    def encode(self, dim=None,
               index=True, feature=True,
               vertical=False):

        if not dim:
            dim = self.index + 1

        x = np.array([])

        if index:
            x = np.concatenate((x, self.index_one_hot(dim)))

        if feature:
            x = np.concatenate((x, self.feature))

        return x if not vertical else np.array([x]).T

    def index_one_hot(self, dim):
        if self.index >= dim:
            raise ValueError('number of dimensions must be greater than index: %d' % self.index)

        x = np.zeros(dim)
        x[self.index] = 1.
        return x


class User(Base):
    pass


class Item(Base):
    pass


class Event:

    def __init__(self, user, item, value, context=np.array([0.])):
        self.user = user
        self.item = item
        self.value = value
        self.context = context

    def encode(self, n_user=None, n_item=None,
               index=True, feature=True, context=True,
               vertical=False):

        x = self.user.encode(dim=n_user, index=index,
                             feature=feature, vertical=False)

        if context:
            x = np.concatenate((x, self.context))

        iv = self.item.encode(dim=n_item, index=index,
                              feature=feature, vertical=False)
        x = np.concatenate((x, iv))

        return x if not vertical else np.array([x]).T
