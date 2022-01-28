"""Utility functions for feature hashing that encodes a feature value to a vector.
"""

import mmh3
import numpy as np


def n_feature_hash(feature, dims, seeds):
    """N-hot-encoded feature hashing.

    Parameters
    ----------
    feature : str
        Target feature value represented as string.

    dims : list of int
        Number of dimensions for each hash value.

    seeds : list of float)
        Seed of each hash function (MurmurHash3).

    Returns
    -------
    array
        n-hot-encoded vector for ``feature``.
    """
    vec = np.zeros(sum(dims))
    offset = 0

    for seed, dim in zip(seeds, dims):
        vec[offset : (offset + dim)] = feature_hash(feature, dim, seed)
        offset += dim

    return vec


def feature_hash(feature, dim, seed=123):
    """Onehot-encoded Feature hashing.

    Parameters
    ----------
    feature : str
        Target feature value represented as string.

    dim : int
        Number of dimensions for a hash value.

    seed : float
        Seed of a MurmurHash3 hash function.

    Returns
    -------
    array
        Onehot-encoded vector for ``feature``.
    """
    vec = np.zeros(dim)
    i = mmh3.hash(feature, seed) % dim
    vec[i] = 1
    return vec


def multiple_feature_hash(feature, dim, seed=123):
    """Onehot-encoded feature hashing using multiple hash functions.
    This technique is effective to prevent collisions.

    Parameters
    ----------
    feature : str
        Target feature value represented as string.

    dim : int
        Number of dimensions for a hash value.

    seed : float
        Seed of a MurmurHash3 hash function.

    Returns
    -------
    array
        Onehot-encoded vector for ``feature``.
    """
    vec = np.zeros(dim)
    i = mmh3.hash(feature, seed) % dim
    vec[i] = 1 if mmh3.hash(feature) % 2 else -1
    return vec
