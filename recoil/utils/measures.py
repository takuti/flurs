import numpy as np


def count_true_positive(truth, recommend):
    """Count number of true positives from given sets of samples.

    Args:
        truth (numpy 1d array): Set of truth samples.
        recommend (numpy 1d array): Ordered set of recommended samples.

    Returns:
        int: Number of true positives.

    """
    tp = 0
    for r in recommend:
        if r in truth:
            tp += 1
    return tp


def recall(truth, recommend, k):
    """Recall@k.

    Args:
        truth (numpy 1d array): Set of truth samples.
        recommend (numpy 1d array): Ordered set of recommended samples.
        k (int): Top-k items in `recommend` will be recommended.

    Returns:
        float: Recall@k.

    """
    return count_true_positive(truth, recommend[:k]) / truth.size


def precision(truth, recommend, k):
    """Precision@k.

    Args:
        truth (numpy 1d array): Set of truth samples.
        recommend (numpy 1d array): Ordered set of recommended samples.
        k (int): Top-k items in `recommend` will be recommended.

    Returns:
        float: Precision@k.

    """
    return count_true_positive(truth, recommend[:k]) / k


def mean_average_precision(truth, recommend):
    """Mean Average Precision (MAP).

    Args:
        truth (numpy 1d array): Set of truth samples.
        recommend (numpy 1d array): Ordered set of recommended samples.

    Returns:
        float: MAP.

    """
    tp = accum = 0
    for n in range(recommend.size):
        if recommend[n] in truth:
            tp += 1
            accum += (tp / (n + 1))
    return accum / truth.size


def auc(truth, recommend):
    """Area under the ROC curve (AUC).

    Args:
        truth (numpy 1d array): Set of truth samples.
        recommend (numpy 1d array): Ordered set of recommended samples.

    Returns:
        float: AUC.

    """
    tp = correct = 0
    for r in recommend:
        if r in truth:
            # keep track number of true positives placed before
            tp += 1
        else:
            correct += tp
    # number of all possible tp-fp pairs
    pairs = tp * (recommend.size - tp)
    return correct / pairs


def mrr(truth, recommend):
    """Mean Reciprocal Rank (MRR).

    Args:
        truth (numpy 1d array): Set of truth samples.
        recommend (numpy 1d array): Ordered set of recommended samples.

    Returns:
        float: MRR.

    """
    for n in range(recommend.size):
        if recommend[n] in truth:
            return 1 / (n + 1)
    return 0


def mpr(truth, recommend):
    """Mean Percentile Rank (MPR).

    Args:
        truth (numpy 1d array): Set of truth samples.
        recommend (numpy 1d array): Ordered set of recommended samples.

    Returns:
        float: MPR.

    """
    accum = 0
    n_recommend = recommend.size
    for t in truth:
        r = np.where(recommend == t)[0][0] / n_recommend
        accum += r
    return accum * 100 / truth.size


def ndcg(truth, recommend, k):
    """Normalized Discounted Cumulative Grain (NDCG).

    Args:
        truth (numpy 1d array): Set of truth samples.
        recommend (numpy 1d array): Ordered set of recommended samples.
        k (int): Top-k items in `recommend` will be recommended.

    Returns:
        float: NDCG.

    """
    dcg = idcg = 0
    for n in range(k):
        d = 1 / np.log2(n + 2)
        if recommend[n] in truth:
            dcg += d
        idcg += d
    return dcg / idcg
