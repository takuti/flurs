from abc import ABCMeta, abstractmethod

from logging import getLogger, StreamHandler, Formatter, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setFormatter(Formatter('[%(process)d] %(message)s'))
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

import numpy as np


class Recommender:

    """Base class for experimentation of the incremental models with positive-only feedback.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, n_item, **params):
        """Set/initialize parameters.

        Args:
            n_item (int): Number of pre-defined items.

        """
        self.n_item = n_item

        # set parameters
        self.params = params

        # initialize models and user/item information
        self.__clear()

    @abstractmethod
    def clear(self):
        """Initialize model parameters and user/item info.

        """
        self.n_user = 0
        self.users = {}
        pass

    @abstractmethod
    def check(self, d):
        """Check if user/item is new.

        For new users/items, append their information into the dictionaries.

        """
        u_index = d['u_index']

        if u_index not in self.users:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

        pass

    @abstractmethod
    def update(self, d, is_batch_train):
        """Update model parameters based on d, a sample represented as a dictionary.

        Args:
            d (dict): A dictionary which has data of a sample.

        """
        pass

    @abstractmethod
    def recommend(self, d, target_i_indices):
        """Recommend items for a user represented as a dictionary d.

        First, scores are computed.
        Next, `self.__scores2recos()` is called to convert the scores into a recommendation list.

        Args:
            d (dict): A dictionary which has data of a sample.
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
