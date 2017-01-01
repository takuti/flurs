import mmh3
import numpy as np


def n_feature_hash(feature, dims, seeds):
    """N-hot-encoded feature hashing.

    Args:
        feature (str): Target feature represented as string.
        dims (list of int): Number of dimensions for each hash value.
        seeds (list of float): Seed of each hash function (mmh3).

    Returns:
        numpy 1d array: n-hot-encoded feature vector for `s`.

    """
    vec = np.zeros(sum(dims))
    offset = 0

    for seed, dim in zip(seeds, dims):
        vec[offset:(offset + dim)] = feature_hash(feature, dim, seed)
        offset += dim

    return vec


def feature_hash(feature, dim, seed=123):
    """Feature hashing.

    Args:
        feature (str): Target feature represented as string.
        dim (int): Number of dimensions for a hash value.
        seed (float): Seed of a MurmurHash3 hash function.

    Returns:
        numpy 1d array: one-hot-encoded feature vector for `s`.

    """
    vec = np.zeros(dim)
    i = mmh3.hash(feature, seed) % dim
    vec[i] = 1
    return vec


def multiple_feature_hash(feature, dim, seed=123):
    """Feature hashing using multiple hash function.
    This technique is effective to prevent collisions.

    Args:
        feature (str): Target feature represented as string.
        dim (int): Number of dimensions for a hash value.
        seed (float): Seed of a MurmurHash3 hash function.

    Returns:
        numpy 1d array: one-hot-encoded feature vector for `s`.

    """
    vec = np.zeros(dim)
    i = mmh3.hash(feature, seed) % dim
    vec[i] = 1 if mmh3.hash(feature) % 2 else -1
    return vec
