"""
============
MovieLens 1M
============

USAGE: %(program)s PATH_TO_MOVIELENS_1M_DIR
"""

from flurs.datasets import fetch_movielens
from flurs.recommender import FMRecommender
from flurs.evaluator import Evaluator

import logging
import os
import sys


if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    if len(sys.argv) < 2:
        print(globals()["__doc__"] % locals())
        sys.exit(1)

    logging.info("converting data into FluRS input object")
    data = fetch_movielens(data_home=sys.argv[1], size="1m")

    logging.info("initialize recommendation model and evaluation module")
    rec = FMRecommender(
        p=sum(data.contexts.values()),  # number of dimensions of input vector
        k=40,
        l2_reg_w0=2.0,
        l2_reg_w=8.0,
        l2_reg_V=16.0,
        learn_rate=0.004,
    )
    rec.initialize()
    evaluator = Evaluator(rec, data.can_repeat)

    n_batch_train = int(data.n_sample * 0.2)  # 20% for pre-training to avoid cold-start
    n_batch_test = int(data.n_sample * 0.1)  # 10% for evaluation of pre-training
    batch_tail = n_batch_train + n_batch_test

    # pre-train
    # 20% for batch training | 10% for batch evaluate
    # after the batch training, 10% samples are used for incremental updating
    logging.info("batch pre-training before streaming input")
    evaluator.fit(
        data.samples[:n_batch_train],
        data.samples[n_batch_train:batch_tail],
        n_epoch=1,  # single pass even for batch training
    )

    # 70% incremental evaluation and updating
    logging.info("incrementally predict, evaluate and update the recommender")
    res = evaluator.evaluate(data.samples[batch_tail:])

    print(res)
