from .base import FeatureRecommenderMixin

import time
import numpy as np

from collections import deque
from . import logger


class Evaluator(object):

    """Base class for experimentation of the incremental models with positive-only feedback.
    """

    def __init__(self, recommender, repeat=True, maxlen=None, debug=False):
        """Set/initialize parameters.

        Args:
            recommender (Recommender): Instance of a recommender which has been initialized.
            repeat (boolean): Choose whether the same item can be repeatedly interacted by the same user.
            maxlen (int): Size of an item buffer which stores most recently observed items.

        """
        self.rec = recommender
        self.feature_rec = issubclass(recommender.__class__, FeatureRecommenderMixin)

        self.repeat = repeat

        # create a ring buffer
        # save items which are observed in most recent `maxlen` events
        self.item_buffer = deque(maxlen=maxlen)

        self.debug = debug

    def fit(self, train_events, test_events, n_epoch=1):
        """Train a model using the first 30% positive events to avoid cold-start.

        Evaluation of this batch training is done by using the next 20% positive events.
        After the batch SGD training, the models are incrementally updated by using the 20% test events.

        Args:
            train_events (list of Event): Positive training events (0-30%).
            test_events (list of Event): Test events (30-50%).
            n_epoch (int): Number of epochs for the batch training.

        """
        # make initial status for batch training
        for e in train_events:
            self.__validate(e)
            self.rec.users[e.user.index]['known_items'].add(e.item.index)
            self.item_buffer.append(e.item.index)

        # for batch evaluation, temporarily save new users info
        for e in test_events:
            self.__validate(e)
            self.item_buffer.append(e.item.index)

        self.__batch_update(train_events, test_events, n_epoch)

        # batch test events are considered as a new observations;
        # the model is incrementally updated based on them before the incremental evaluation step
        for e in test_events:
            self.rec.users[e.user.index]['known_items'].add(e.item.index)
            self.rec.update(e)

    def evaluate(self, test_events):
        """Iterate recommend/update procedure and compute incremental recall.

        Args:
            test_events (list of Event): Positive test events.

        Returns:
            list of tuples: (rank, recommend time, update time)

        """
        for i, e in enumerate(test_events):
            self.__validate(e)

            # target items (all or unobserved depending on a detaset)
            unobserved = set(self.item_buffer)
            if not self.repeat:
                unobserved -= self.rec.users[e.user.index]['known_items']

            # item i interacted by user u must be in the recommendation candidate
            # even if it is a new item
            unobserved.add(e.item.index)

            candidates = np.asarray(list(unobserved))

            # make top-{at} recommendation for the 1001 items
            start = time.perf_counter()
            recos, scores = self.__recommend(e, candidates)
            recommend_time = (time.perf_counter() - start)

            rank = np.where(recos == e.item.index)[0][0]

            # Step 2: update the model with the observed event
            self.rec.users[e.user.index]['known_items'].add(e.item.index)
            start = time.perf_counter()
            self.rec.update(e)
            update_time = (time.perf_counter() - start)

            self.item_buffer.append(e.item.index)

            # (top-1 score, where the correct item is ranked, rec time, update time)
            yield scores[0], rank, recommend_time, update_time

    def __recommend(self, e, candidates):
        if self.feature_rec:
            return self.rec.recommend(e.user, candidates, e.context)
        else:
            return self.rec.recommend(e.user, candidates)

    def __validate(self, e):
        self.__validate_user(e)
        self.__validate_item(e)

    def __validate_user(self, e):
        if self.rec.is_new_user(e.user.index):
            self.rec.register_user(e.user)

    def __validate_item(self, e):
        if self.rec.is_new_item(e.item.index):
            self.rec.register_item(e.item)

    def __batch_update(self, train_events, test_events, n_epoch):
        """Batch update called by the fitting method.

        Args:
            train_events (list of Event): Positive training events.
            test_events (list of Event): Test events.
            n_epoch (int): Number of epochs for the batch training.

        """
        for epoch in range(n_epoch):
            # SGD requires us to shuffle events in each iteration
            # * if n_epoch == 1
            #   => shuffle is not required because it is a deterministic training (i.e. matrix sketching)
            if n_epoch != 1:
                np.random.shuffle(train_events)

            # train
            for e in train_events:
                self.rec.update(e, batch_train=True)

            # test
            MPR = self.__batch_evaluate(test_events)
            if self.debug:
                logger.debug('epoch %2d: MPR = %f' % (epoch + 1, MPR))

    def __batch_evaluate(self, test_events):
        """Evaluate the current model by using the given test events.

        Args:
            test_events (list of Event): Current model is evaluated by these events.

        Returns:
            float: Mean Percentile Rank for the test set.

        """
        percentiles = np.zeros(len(test_events))

        all_items = set(self.item_buffer)
        for i, e in enumerate(test_events):

            # check if the data allows users to interact the same items repeatedly
            unobserved = all_items
            if not self.repeat:
                # make recommendation for all unobserved items
                unobserved -= self.rec.users[e.user.index]['known_items']
                # true item itself must be in the recommendation candidates
                unobserved.add(e.item.index)

            candidates = np.asarray(list(unobserved))
            recos, scores = self.__recommend(e, candidates)

            pos = np.where(recos == e.item.index)[0][0]
            percentiles[i] = pos / (len(recos) - 1) * 100

        return np.mean(percentiles)
