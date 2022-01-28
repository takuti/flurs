from .data.entity import User, Item

import numpy as np


class RecommenderMixin(object):

    """Mixin injected into a model to make it a recommender."""

    def initialize(self, *args):
        """Initialize a recommender by resetting stored users and items.
        Default the number of users and items to zero.
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
        """Check if a user is already registered to a recommender.

        Parameters
        ----------
        u : int
            User index.

        Returns
        -------
        bool
            Whether the user is new.
        """
        return u not in self.users

    def register(self, actor):
        """Register a user or item to a recommender.
        Delegate the process to `register_user` or `register_item`.

        Parameters
        ----------
        actor : User or Item
            A `User` or `Item` instance to register.
        """
        t = type(actor)
        if t == User:
            self.register_user(actor)
        elif t == Item:
            self.register_item(actor)

    def register_user(self, user):
        """Register a user to a recommender with an empty dictionary that
        records a set of item indices observed in the past.

        Parameters
        ----------
        user : User
            A `User` instance to register.
        """
        self.users[user.index] = {"known_items": set()}
        self.n_user += 1

    def is_new_item(self, i):
        """Check if an item is already registered to a recommender.

        Parameters
        ----------
        i : int
            Item index.

        Returns
        -------
        bool
            Whether the item is new.
        """
        return i not in self.items

    def register_item(self, item):
        """Register an item to a recommender with an empty dictionary that is
        potentially used to record any auxiliary information.

        Parameters
        ----------
        item : Item
            A `Item` instance to register.
        """
        self.items[item.index] = {}
        self.n_item += 1

    def update(self, e, batch_train):
        """Update model parameters given a single user-item interaction.

        Parameters
        ----------
        e : Event
            Observed event representing a user-item interaction.

        batch_train : bool
            Let recommender know if the update operation is part of batch training
            rather than incremental, online update.
        """
        pass

    def score(self, user, candidates):
        """Compute scores for the pairs of given user and item candidates.

        Parameters
        ----------
        user : User
            Target user.

        candidates : numpy array, (# candidates, )
            Target item indices.

        Returns
        -------
        array, (# candidates, )
            Predicted values for the given user-candidates pairs.
        """
        return

    def recommend(self, user, candidates):
        """Recommend items for a user by calculating sorted list of item candidates.

        1. Scores are computed.
        2. `scores2recos()` is called to convert the scores into a recommendation list.

        Parameters
        ----------
        user : User
            Target user.

        candidates : numpy array, (# target items, )
            Target items indices.
            Only these items are considered as the recommendation candidates.

        Returns
        -------
        (array, array)
            A tuple of ``(sorted list of item indices, sorted scores)``.
        """
        return

    def scores2recos(self, scores, candidates, rev=False):
        """Get recommendation list for a user based on scores.

        Parameters
        ----------
        scores : numpy array, (n_target_items,)
            Scores for the target items. Smaller score indicates a promising item.

        candidates : numpy array, (# target items, )
            Target items indices.
            Only these items are considered as the recommendation candidates.

        rev : bool, default=False
            If ``True``, sort and return items in an descending order.
            The default is an ascending order (i.e., smaller scores are more promising).

        Returns
        -------
        (array, array)
            A tuple of ``(sorted list of item indices, sorted scores)``.
        """
        sorted_indices = np.argsort(scores)

        if rev:
            sorted_indices = sorted_indices[::-1]

        return candidates[sorted_indices], scores[sorted_indices]


class FeatureRecommenderMixin(RecommenderMixin):

    """Mixin injected into a model to make it a feature-based recommender."""

    def score(self, user, candidates, context):
        """Compute scores for the pairs of given user and item candidates.

        Parameters
        ----------
        user : User
            Target user.

        candidates : numpy array, (# candidates, )
            Target item indices.

        context : (numpy array)
            Feature vector representing contextual information.

        Returns
        -------
        array, (# candidates, )
            Predicted values for the given user-candidates pairs.
        """
        return

    def recommend(self, user, candidates, context):
        """Recommend items for a user by calculating sorted list of item candidates.

        1. Scores are computed.
        2. `scores2recos()` is called to convert the scores into a recommendation list.

        Parameters
        ----------
        user : User
            Target user.

        candidates : numpy array, (# target items, )
            Target items indices.
            Only these items are considered as the recommendation candidates.

        context : (numpy array)
            Feature vector representing contextual information.

        Returns
        -------
        (array, array)
            A tuple of ``(sorted list of item indices, sorted scores)``.
        """
        return
