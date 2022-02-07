"""Ranking-based evaluation metrics for recommender systems"""

import numpy as np


def count_true_positive(truth, recommend):
    """Count number of true positives from given sets of samples.

    Parameters
    ----------
    truth : numpy 1d array
        Set of truth samples.

    recommend : numpy 1d array
        Ordered listed of recommended samples.

    Returns
    -------
    int
        Number of true positives.
    """
    tp = 0
    for r in recommend:
        if r in truth:
            tp += 1
    return tp


def recall(truth, recommend, k=None):
    """Recall@k.

    Parameters
    ----------
    truth : numpy 1d array
        Set of truth samples.

    recommend : numpy 1d array
        Ordered listed of recommended samples.

    k : int or None, default=None
        Top-k items in ``recommend`` are considered to be recommended.
        Defaults to ``len(recommend)``.

    Returns
    -------
    float
        Recall@k.
    """
    if len(truth) == 0:
        if len(recommend) == 0:
            return 1.0
        return 0.0

    if k is None:
        k = len(recommend)
    return count_true_positive(truth, recommend[:k]) / float(truth.size)


def precision(truth, recommend, k=None):
    """Precision@k.

    Parameters
    ----------
    truth : numpy 1d array
        Set of truth samples.

    recommend : numpy 1d array
        Ordered listed of recommended samples.

    k : int or None, default=None
        Top-k items in ``recommend`` are considered to be recommended.
        Defaults to ``len(recommend)``.

    Returns
    -------
    float
        Precision@k.
    """
    if len(recommend) == 0:
        if len(truth) == 0:
            return 1.0
        return 0.0

    if k is None:
        k = len(recommend)
    return count_true_positive(truth, recommend[:k]) / float(k)


def average_precision(truth, recommend):
    """Average Precision (AP).

    Parameters
    ----------
    truth : numpy 1d array
        Set of truth samples.

    recommend : numpy 1d array
        Ordered listed of recommended samples.

    Returns
    -------
    float
        Average Precision.
    """
    if len(truth) == 0:
        if len(recommend) == 0:
            return 1.0
        return 0.0

    tp = accum = 0.0
    for n in range(recommend.size):
        if recommend[n] in truth:
            tp += 1.0
            accum += tp / (n + 1.0)
    return accum / truth.size


def auc(truth, recommend):
    """Area under the ROC curve (AUC).

    Parameters
    ----------
    truth : numpy 1d array
        Set of truth samples.

    recommend : numpy 1d array
        Ordered listed of recommended samples.

    Returns
    -------
    float
        AUC.
    """
    tp = correct = 0.0
    for r in recommend:
        if r in truth:
            # keep track number of true positives placed before
            tp += 1.0
        else:
            correct += tp
    # number of all possible tp-fp pairs
    pairs = tp * (recommend.size - tp)

    # if there is no TP (or no FP), it's meaningless for this metric (i.e., AUC=0.5)
    if pairs == 0:
        return 0.5

    return correct / pairs


def reciprocal_rank(truth, recommend):
    """Reciprocal Rank (RR).

    Parameters
    ----------
    truth : numpy 1d array
        Set of truth samples.

    recommend : numpy 1d array
        Ordered listed of recommended samples.

    Returns
    -------
    float
        Reciprocal Rank.
    """
    for n in range(recommend.size):
        if recommend[n] in truth:
            return 1.0 / (n + 1)
    return 0.0


def mpr(truth, recommend):
    """Mean Percentile Rank (MPR).

    Parameters
    ----------
    truth : numpy 1d array
        Set of truth samples.

    recommend : numpy 1d array
        Ordered listed of recommended samples.

    Returns
    -------
    float
        Mean Percentile Rank.
    """
    if len(recommend) == 0 and len(truth) == 0:
        return 0.0  # best
    elif len(truth) == 0 or len(truth) == 0:
        return 100.0  # worst

    accum = 0.0
    n_recommend = recommend.size
    for t in truth:
        r = np.where(recommend == t)[0][0] / float(n_recommend)
        accum += r
    return accum * 100.0 / truth.size


def ndcg(truth, recommend, k=None):
    """Normalized Discounted Cumulative Grain (NDCG).

    Parameters
    ----------
    truth : numpy 1d array
        Set of truth samples.

    recommend : numpy 1d array
        Ordered listed of recommended samples.

    k : int or None, default=None
        Top-k items in ``recommend`` are considered to be recommended.
        Defaults to ``len(recommend)``.

    Returns
    -------
    float
        NDCG@k.
    """
    if k is None:
        k = len(recommend)

    def idcg(n_possible_truth):
        res = 0.0
        for n in range(n_possible_truth):
            res += 1.0 / np.log2(n + 2)
        return res

    dcg = 0.0
    for n, r in enumerate(recommend[:k]):
        if r not in truth:
            continue
        dcg += 1.0 / np.log2(n + 2)

    res_idcg = idcg(np.min([truth.size, k]))
    if res_idcg == 0.0:
        return 0.0
    return dcg / res_idcg
