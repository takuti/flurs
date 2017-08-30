from .data.entity import User, Item

import numpy as np


class RecommenderMixin(object):

    """Mixin injected into a model to make it a recommender.
    """

    def initialize(self, *args):
        """Initialize a recommender by resetting stored users and items.
        """
        # number of observed users
        self.n_user = 0

        # store user data
        self.users = {}

        # number of observed items
        self.n_item = 0

        # store item data
        self.items = {}

    def is_new_user(self, u):
        """Check if user is new.

        Args:
            u (int): User index.

        Returns:
            boolean: Whether the user is new.

        """
        return u not in self.users

    def register(self, entity):
        t = type(entity)
        if t == User:
            self.register_user(entity)
        elif t == Item:
            self.register_item(entity)

    def register_user(self, user):
        """For new users, append their information into the dictionaries.

        Args:
            user (User): User.

        """
        self.users[user.index] = {'known_items': set()}
        self.n_user += 1

    def is_new_item(self, i):
        """Check if item is new.

        Args:
            i (int): Item index.

        Returns:
            boolean: Whether the item is new.

        """
        return i not in self.items

    def register_item(self, item):
        """For new items, append their information into the dictionaries.

        Args:
            item (Item): Item.

        """
        self.items[item.index] = {}
        self.n_item += 1

    def update(self, e, batch_train):
        """Update model parameters based on d, a sample represented as a dictionary.

        Args:
            e (Event): Observed event.

        """
        pass

    def score(self, user, candidates):
        """Compute scores for the pairs of given user and item candidates.

        Args:
            user (User): Target user.
            candidates (numpy array; (# candidates, )): Target item' indices.

        Returns:
            numpy float array; (# candidates, ): Predicted values for the given user-candidates pairs.

        """
        return

    def recommend(self, user, candidates):
        """Recommend items for a user represented as a dictionary d.

        First, scores are computed.
        Next, `self.__scores2recos()` is called to convert the scores into a recommendation list.

        Args:
            user (User): Target user.
            candidates (numpy array; (# target items, )): Target items' indices. Only these items are considered as the recommendation candidates.

        Returns:
            (numpy array, numpy array) : (Sorted list of items, Sorted scores).

        """
        return

    def scores2recos(self, scores, candidates, rev=False):
        """Get recommendation list for a user u_index based on scores.

        Args:
            scores (numpy array; (n_target_items,)):
                Scores for the target items. Smaller score indicates a promising item.
            candidates (numpy array; (# target items, )): Target items' indices. Only these items are considered as the recommendation candidates.
            rev (bool): If true, return items in an descending order. A ascending order (i.e., smaller scores are more promising) is default.

        Returns:
            (numpy array, numpy array) : (Sorted list of items, Sorted scores).

        """
        sorted_indices = np.argsort(scores)

        if rev:
            sorted_indices = sorted_indices[::-1]

        return candidates[sorted_indices], scores[sorted_indices]


class FeatureRecommenderMixin(RecommenderMixin):

    """Mixin injected into a model to make it a feature-based recommender.
    """

    def score(self, user, candidates, context):
        """Compute scores for the pairs of given user and item candidates.

        Args:
            user (User): Target user.
            candidates (numpy array; (# candidates, )): Target item' indices.
            context (numpy 1d array): Feature vector representing contextual information.

        Returns:
            numpy float array; (# candidates, ): Predicted values for the given user-candidates pairs.

        """
        return

    def recommend(self, user, candidates, context):
        """Recommend items for a user represented as a dictionary d.

        First, scores are computed.
        Next, `self.__scores2recos()` is called to convert the scores into a recommendation list.

        Args:
            user (User): Target user.
            candidates (numpy array; (# target items, )): Target items' indices. Only these items are considered as the recommendation candidates.
            context (numpy 1d array): Feature vector representing contextual information.

        Returns:
            (numpy array, numpy array) : (Sorted list of items, Sorted scores).

        """
        return
