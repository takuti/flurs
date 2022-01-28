import numpy as np


class BaseActor(object):
    def __init__(self, index, feature=np.array([])):
        self.index = index
        self.feature = feature

    def encode(self, dim=None, index=True, feature=True, vertical=False):
        """Encode an actor to an input vector for feature-based recommenders.

        Parameters
        ----------
        dim : int or None, default=None
            Number of dimensions for onehot-encoded index.
            Use ``self.index + 1`` if ``None``.

        index : bool, default=True
            Include onehot-encoded index to an input vector.

        feature : bool, default=True
            Include features associated with an actor.

        vertical : bool, default=False
            Return as a transposed n-by-1 vertical vector.

        Returns
        -------
        array
            n-dimensional vector representing an actor.
            Size can be ``(n, 1)`` or ``(1, n)``, depending on ``vertical`` parameter.

        """

        if not dim:
            dim = self.index + 1

        x = np.array([])

        if index:
            x = np.concatenate((x, self.index_one_hot(dim)))

        if feature:
            x = np.concatenate((x, self.feature))

        return x if not vertical else np.array([x]).T

    def index_one_hot(self, dim):
        """Onehot-encode own index to a vector.

        Parameters
        ----------
        dim : int
            Number of dimensions of an output vector.
            Must be greater than or equal to `self.index`.

        Returns
        -------
        array
            ``dim``-dimensional onehot-encoded vector.

        """
        if self.index >= dim:
            raise ValueError(
                "number of dimensions must be greater than index: %d" % self.index
            )

        x = np.zeros(dim)
        x[self.index] = 1.0
        return x


class User(BaseActor):

    """User object for recommenders.

    Parameters
    ----------
    index : int
        User index used as their ID. Starting from ``0``.

    features : numpy array, default=empty
        Feature vector associated with a user.
        An element can be age, gender, days from last access, etc.

    """

    def __str__(self):
        if len(self.feature) == 0:
            return "User(index={})".format(self.index)
        else:
            return "User(index={}, feature={})".format(self.index, self.feature)


class Item(BaseActor):

    """Item object for recommenders.

    Parameters
    ----------
    index : int
        Item index used as their ID. Starting from ``0``.

    features : numpy array, default=empty
        Feature vector associated with an item.
        An element can be price, category, date published, etc.

    """

    def __str__(self):
        if len(self.feature) == 0:
            return "Item(index={})".format(self.index)
        else:
            return "Item(index={}, feature={})".format(self.index, self.feature)


class Event(object):

    """An event object that represents a single user-item interaction.

    Parameters
    ----------
    user : User
        A `User` instance.

    item : Item
        An `Item` instance.

    value : float, default=1.0
        A value representing the feedback. ``1.0`` in case of positive-only feedback,
        or 5-scale value for rating prediction, for example.

    context : numpy array, default=empty
        Vector-represented contextual information associated with an interaction.
        An element can be day of the week, time, weather, etc.

    """

    def __init__(self, user, item, value=1.0, context=np.array([])):
        self.user = user
        self.item = item
        self.value = value
        self.context = context

    def encode(
        self,
        n_user=None,
        n_item=None,
        index=True,
        feature=True,
        context=True,
        vertical=False,
    ):
        """Encode an event to an input vector for feature-based recommenders.

        Parameters
        ----------
        n_user : int
            Number of users currently registered to a recommender.

        n_item : int
            Number of items currently registered to a recommender.

        index : bool, default=True
            Include onehot-encoded user/item index to an input vector.

        feature : bool, default=True
            Include features associated with user and item.

        context : bool, default=True
            Include event-specific contextual information to a vector.

        vertical : bool, default=False
            Return as a transposed n-by-1 vertical vector.

        Returns
        -------
        array
            n-dimensional vector representing a user-item interaction.
            Size can be ``(n, 1)`` or ``(1, n)``, depending on ``vertical`` parameter.

        """

        x = self.user.encode(dim=n_user, index=index, feature=feature, vertical=False)

        if context:
            x = np.concatenate((x, self.context))

        iv = self.item.encode(dim=n_item, index=index, feature=feature, vertical=False)
        x = np.concatenate((x, iv))

        assert len(x) > 0, "feature vector has zero dimension"

        return x if not vertical else np.array([x]).T

    def __str__(self):
        if len(self.context) == 0:
            return "Event(user={}, item={}, value={})".format(
                self.user, self.item, self.value
            )
        else:
            return "Event(user={}, item={}, value={}, context={})".format(
                self.user, self.item, self.value, self.context
            )
