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
    def __init__(self, **params):
        """Set/initialize parameters.

        """
        pass

    @abstractmethod
    def clear(self):
        """Initialize model parameters and user/item info.

        """
        pass

    @abstractmethod
    def check(self, u, i, u_feature, i_feature):
        """Check if user/item is new.

        For new users/items, append their information into the dictionaries.

        Args:
            u (int): User index.
            i (int): Item index.

        Returns:
            (boolean, boolean) : (whether user is new, whether item is new)

        """
        return

    @abstractmethod
    def update(self, u, i, r, context, is_batch_train):
        """Update model parameters based on d, a sample represented as a dictionary.

        Args:
            u (int): User index.
            i (int): Item index.
            r (float): Observed true value.

        """
        pass

    @abstractmethod
    def recommend(self, u, target_i_indices, context):
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
