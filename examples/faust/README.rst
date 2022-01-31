Streaming Recommendation with Faust
-----------------------------------

`Faust <https://faust.readthedocs.io/en/latest/>`_ is a streaming processing library in Python.

.. code:: sh

    brew services start zookeeper
    brew services start kafka

.. code:: sh

    faust -A recommender worker -l info

.. code:: sh

    python producer.py /path/to/ml-100k/u.data
