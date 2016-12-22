import mmh3
import numpy as np


def n_feature_hash(s, seeds, dims):
    """n-hot-encoded feature hashing.

    Args:
        s (str): Target string.
        seeds (list of float): Seed of each hash function (mmh3).
        dims (list of int): Number of dimensions for each hash value.

    Returns:
        numpy 1d array: n-hot-encoded feature vector for `s`.

    """
    vec = np.zeros(sum(dims))
    offset = 0

    for seed, dim in zip(seeds, dims):
        i = mmh3.hash(s, seed) % dim
        vec[offset + i] = 1
        offset += dim

    return vec
