from abc import ABCMeta, abstractmethod

import numpy as np


class Recommender:

    """Base class for experimentation of the incremental models with positive-only feedback.

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        # number of observed users
        self.n_user = 0

        # store user data
        self.users = {}

        # number of observed items
        self.n_item = 0

        # store item data
        self.items = {}

        # initialize model parameters
        self.init_model()

    @abstractmethod
    def init_model(self):
        """Initialize model parameters.

        """
        pass

    def is_new_user(self, u):
        """Check if user is new.

        Args:
            u (int): User index.

        Returns:
            boolean: whether user is new

        """
        return u not in self.users

    @abstractmethod
    def add_user(self, u):
        """For new users, append their information into the dictionaries.

        Args:
            u (int): User index.

        """
        self.users[u] = {'observed': set()}
        self.n_user += 1

    def is_new_item(self, i):
        """Check if item is new.

        Args:
            i (int): Item index.

        Returns:
            boolean: whether item is new

        """
        return i not in self.items

    @abstractmethod
    def add_item(self, i):
        """For new items, append their information into the dictionaries.

        Args:
            i (int): Item index.

        """
        self.items[i] = {}
        self.n_item += 1

    @abstractmethod
    def update(self, u, i, r, is_batch_train):
        """Update model parameters based on d, a sample represented as a dictionary.

        Args:
            u (int): User index.
            i (int): Item index.
            r (float): Observed true value.

        """
        pass

    @abstractmethod
    def recommend(self, u, target_i_indices):
        """Recommend items for a user represented as a dictionary d.

        First, scores are computed.
        Next, `self.__scores2recos()` is called to convert the scores into a recommendation list.

        Args:
            u (int): Target user index.
            target_i_indices (numpy array; (# target items, )): Target items' indices. Only these items are considered as the recommendation candidates.

        Returns:
            (numpy array, numpy array) : (Sorted list of items, Sorted scores).

        """
        return

    def __scores2recos(self, scores, target_i_indices):
        """Get recommendation list for a user u_index based on scores.

        Args:
            scores (numpy array; (n_target_items,)):
                Scores for the target items. Smaller score indicates a promising item.
            target_i_indices (numpy array; (# target items, )): Target items' indices. Only these items are considered as the recommendation candidates.

        Returns:
            (numpy array, numpy array) : (Sorted list of items, Sorted scores).

        """
        sorted_indices = np.argsort(scores)
        return target_i_indices[sorted_indices], scores[sorted_indices]
