API Reference
=============

.. currentmodule:: flurs

Data representation
-------------------

.. autosummary::
    :toctree: generated

    types.User
    types.Item
    types.Event
    datasets.movielens

Baseline recommenders
---------------------

.. autosummary::
    :toctree: generated

    baseline.Random
    baseline.Popular

Collaborative filtering
-----------------------

.. autosummary::
    :toctree: generated

    recommender.UserKNNRecommender
    recommender.MFRecommender
    recommender.BPRMFRecommender

Feature-based recommenders
--------------------------

.. autosummary::
    :toctree: generated

    recommender.FMRecommender
    recommender.SketchRecommender

Vector manipulation utilities
-----------------------------

.. autosummary::
    :toctree: generated

    utils.feature_hash
    utils.projection

Evaluation utilities
--------------------

.. autosummary::
    :toctree: generated

    metrics
    evaluation.Evaluator
