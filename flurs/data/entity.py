import numpy as np


class Base(object):

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

    def __str__(self):
        if len(self.feature) == 1 and self.feature[0] == 0.:
            return 'User(index={})'.format(self.index)
        else:
            return 'User(index={}, feature={})'.format(self.index, self.feature)


class Item(Base):

    def __str__(self):
        if len(self.feature) == 1 and self.feature[0] == 0.:
            return 'Item(index={})'.format(self.index)
        else:
            return 'Item(index={}, feature={})'.format(self.index, self.feature)


class Event(object):

    def __init__(self, user, item, value=1., context=np.array([0.])):
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

    def __str__(self):
        if len(self.context) == 1 and self.context[0] == 0.:
            return 'Event(user={}, item={}, value={})'.format(self.user, self.item, self.value)
        else:
            return 'Event(user={}, item={}, value={}, context={})'.format(self.user, self.item, self.value, self.context)
