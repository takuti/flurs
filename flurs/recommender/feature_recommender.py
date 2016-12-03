from flurs.recommender import recommender

from abc import ABCMeta, abstractmethod


class FeatureRecommender(recommender.Recommender):

    """Base class for experimentation of the incremental models with positive-only feedback.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def recommend(self, user, target_i_indices, context):
        """Recommend items for a user represented as a dictionary d.

        First, scores are computed.
        Next, `self.__scores2recos()` is called to convert the scores into a recommendation list.

        Args:
            user (User): Target user.
            target_i_indices (numpy array; (# target items, )): Target items' indices. Only these items are considered as the recommendation candidates.
            context (numpy 1d array): Feature vector representing contextual information.

        Returns:
            (numpy array, numpy array) : (Sorted list of items, Sorted scores).

        """
        return
