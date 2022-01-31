User Guide
==========

You first need to convert your data to ``User``, ``Item`` and ``Event``:

.. testcode::

  from flurs.data.entity import User, Item, Event

  # define a user with index 0
  user = User(0)

  # define an item with index 0
  item = Item(0)

  # interaction between a user and item
  event = Event(user, item)

Eventually, time-stamped data can be represented as a list of ``Event`` on FluRS.

If you want to use a feature-based recommender (e.g., factorization machines), the entities take additional arguments:

.. testcode::

  import numpy as np

  user = User(0, feature=np.array([0,0,1]))
  item = Item(0, feature=np.array([2,1,1]))
  event = Event(user, item, context=np.array([0,4,0]))

To give an example, a matrix-factorization-based recommender can be used as follows:

.. testcode::

  from flurs.recommender import MFRecommender

  recommender = MFRecommender(k=40)

  recommender.initialize()

  user = User(0)
  recommender.register(user)

  item = Item(0)
  recommender.register(item)

  event = Event(user, item)
  recommender.update(event)

  # specify target user and list of item candidates
  recommender.recommend(user, np.array([0]))
  # => (sorted candidates, scores)
