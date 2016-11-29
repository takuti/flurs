from logging import getLogger, StreamHandler, Formatter, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setFormatter(Formatter('[%(process)d] %(message)s'))
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

import time
import numpy as np


class Evaluator:

    """Base class for experimentation of the incremental models with positive-only feedback.

    """

    def __init__(self, recommender):
        """Set/initialize parameters.

        Args:
            recommender (Recommender): Instance of a recommender.

        """
        self.recommender = recommender

        # initialize models and user/item information
        recommender.init_model()

    def set_can_repeat(self, can_repeat):
        self.can_repeat = can_repeat

    def fit(self, train_samples, test_samples, n_epoch=1):
        """Train a model using the first 30% positive samples to avoid cold-start.

        Evaluation of this batch training is done by using the next 20% positive samples.
        After the batch SGD training, the models are incrementally updated by using the 20% test samples.

        Args:
            train_samples (list of dict): Positive training samples (0-30%).
            test_sample (list of dict): Test samples (30-50%).
            n_epoch (int): Number of epochs for the batch training.

        """
        self.recommender.init_model()

        # make initial status for batch training
        for d in train_samples:
            self.recommender.check(d)
            self.recommender.users[d['u_index']]['observed'].add(d['i_index'])

        # for batch evaluation, temporarily save new users info
        for d in test_samples:
            self.recommender.check(d)

        self.batch_update(train_samples, test_samples, n_epoch)

        # batch test samples are considered as a new observations;
        # the model is incrementally updated based on them before the incremental evaluation step
        for d in test_samples:
            self.recommender.users[d['u_index']]['observed'].add(d['i_index'])
            self.recommender.update(d)

    def batch_update(self, train_samples, test_samples, n_epoch):
        """Batch update called by the fitting method.

        Args:
            train_samples (list of dict): Positive training samples (0-20%).
            test_sample (list of dict): Test samples (20-30%).
            n_epoch (int): Number of epochs for the batch training.

        """
        for epoch in range(n_epoch):
            # SGD requires us to shuffle samples in each iteration
            # * if n_epoch == 1
            #   => shuffle is not required because it is a deterministic training (i.e. matrix sketching)
            if n_epoch != 1:
                np.random.shuffle(train_samples)

            # 20%: update models
            for d in train_samples:
                self.recommender.update(d, is_batch_train=True)

            # 10%: evaluate the current model
            MPR = self.batch_evaluate(test_samples)
            logger.debug('epoch %2d: MPR = %f' % (epoch + 1, MPR))

    def batch_evaluate(self, test_samples):
        """Evaluate the current model by using the given test samples.

        Args:
            test_samples (list of dict): Current model is evaluated by these samples.

        Returns:
            float: Mean Percentile Rank for the test set.

        """
        percentiles = np.zeros(len(test_samples))

        all_items = set(range(self.recommender.n_item))
        for i, d in enumerate(test_samples):

            # check if the data allows users to interact the same items repeatedly
            unobserved = all_items
            if not self.can_repeat:
                # make recommendation for all unobserved items
                unobserved -= self.recommender.users[d['u_index']]['observed']
                # true item itself must be in the recommendation candidates
                unobserved.add(d['i_index'])

            target_i_indices = np.asarray(list(unobserved))
            recos, scores = self.recommender.recommend(d, target_i_indices)

            pos = np.where(recos == d['i_index'])[0][0]
            percentiles[i] = pos / (len(recos) - 1) * 100

        return np.mean(percentiles)

    def evaluate(self, test_samples):
        """Iterate recommend/update procedure and compute incremental recall.

        Args:
            test_samples (list of dict): Positive test samples.

        Returns:
            list of tuples: (rank, recommend time, update time)

        """
        for i, d in enumerate(test_samples):
            self.recommender.check(d)

            u_index = d['u_index']
            i_index = d['i_index']

            # target items (all or unobserved depending on a detaset)
            unobserved = set(range(self.recommender.n_item))
            if not self.can_repeat:
                unobserved -= self.recommender.users[u_index]['observed']
                # * item i interacted by user u must be in the recommendation candidate
                unobserved.add(i_index)
            target_i_indices = np.asarray(list(unobserved))

            # make top-{at} recommendation for the 1001 items
            start = time.clock()
            recos, scores = self.recommender.recommend(d, target_i_indices)
            recommend_time = (time.clock() - start)

            rank = np.where(recos == i_index)[0][0]

            # Step 2: update the model with the observed event
            self.recommender.users[u_index]['observed'].add(i_index)
            start = time.clock()
            self.recommender.update(d)
            update_time = (time.clock() - start)

            # (top-1 score, where the correct item is ranked, rec time, update time)
            yield scores[0], rank, recommend_time, update_time
