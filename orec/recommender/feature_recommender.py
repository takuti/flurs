from orec.recommender.recommender import Recommender

from abc import ABCMeta, abstractmethod

from logging import getLogger, StreamHandler, Formatter, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setFormatter(Formatter('[%(process)d] %(message)s'))
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


class FeatureRecommender(Recommender):

    """Base class for experimentation of the incremental models with positive-only feedback.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def check(self, u, i, u_feature, i_feature):
        """Check if user/item is new.

        For new users/items, append their information into the dictionaries.

        Args:
            u (int): User index.
            i (int): Item index.
            u_feature (numpy 1d array): Feature vector for user.
            i_feature (numpy 1d array): Feature vector for item.

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
            context (numpy 1d array): Feature vector representing contextual information.

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
            context (numpy 1d array): Feature vector representing contextual information.

        Returns:
            (numpy array, numpy array) : (Sorted list of items, Sorted scores).

        """
        return
