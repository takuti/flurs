import numpy as np


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
