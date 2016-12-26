import mmh3
import numpy as np


def n_feature_hash(s, seeds, dims):
    """N-hot-encoded feature hashing.

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
        vec[offset:(offset + dim)] = feature_hash(s, seed, dim)
        offset += dim

    return vec


def feature_hash(s, seed, dim):
    """Feature hashing.

    Args:
        s (str): Target string.
        seed (float): Seed of a MurmurHash3 hash function.
        dim (int): Number of dimensions for a hash value.

    Returns:
        numpy 1d array: one-hot-encoded feature vector for `s`.

    """
    vec = np.zeros(dim)
    i = mmh3.hash(s, seed) % dim
    vec[i] = 1
    return vec
