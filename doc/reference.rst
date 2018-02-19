References
==========

Baseline
--------

.. autofunction:: flurs.baseline.Random

.. autofunction:: flurs.baseline.Popular

Collaborative filtering
-----------------------

.. autofunction:: flurs.recommender.UserKNNRecommender

* M. Pepagelis et al. **Incremental Collaborative Filtering for Highly-Scalable Recommendation Algorithms**. In *Foundations of Intelligent Systems*, pp. 553–561, Springer Berlin Heidelberg, 2005.

.. autofunction:: flurs.recommender.MFRecommender

* J. Vinagre et al. `Fast Incremental Matrix Factorization for Recommendation with Positive-only Feedback <http://link.springer.com/chapter/10.1007/978-3-319-08786-3_41>`_. In *Proc. of UMAP 2014*, pp. 459–470, July 2014.

.. autofunction:: flurs.recommender.BPRMFRecommender

* S. Rendle et al. **BPR: Bayesian Personalized Ranking from Implicit Feedback**. In *Proc. of UAI 2009*, pp. 452–461, June 2009.

Feature-based recommender
-------------------------

.. autofunction:: flurs.recommender.FMRecommender

* T. Kitazawa. `Incremental Factorization Machines for Persistently Cold-Starting Online Item Recommendation <https://arxiv.org/abs/1607.02858>`_. arXiv:1607.02858 [cs.LG], July 2016.

.. autofunction:: flurs.recommender.SketchRecommender

* T. Kitazawa. `Sketching Dynamic User-Item Interactions for Online Item Recommendation <http://dl.acm.org/citation.cfm?id=3022152>`_. In *Proc. of CHIIR 2017*, March 2017.
