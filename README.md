FluRS
===

[![Build Status](https://travis-ci.org/takuti/flurs.svg?branch=master)](https://travis-ci.org/takuti/flurs) [![PyPI version](https://badge.fury.io/py/flurs.svg)](https://badge.fury.io/py/flurs)

***FluRS*** is a Python library for online item recommendation. The name indicates *Flu-** (Flux, Fluid, Fluent) *Recommender Systems* which incrementally adapt to dynamic user-item interactions in a streaming environment.

You can refer to [my article](https://takuti.me/note/flurs/) and [slides](https://speakerdeck.com/takuti/flurs-a-library-for-streaming-recommendation-algorithms) for mode detailed explanation:

[![structure](doc/images/structure.png)](https://speakerdeck.com/takuti/flurs-a-library-for-streaming-recommendation-algorithms)

## Installation

```
$ pip install flurs
```

## Usage

You first need to convert your data to `User`, `Item` and `Event`:

```python
from flurs.data.entity import User, Item, Event

# define a user with index 0
user = User(0)

# define an item with index 0
item = Item(0)

# interaction between a user and item
event = Event(user, item)
```

Eventually, time-stamped data can be represented as a list of `Event` on FluRS.

If you want to use a feature-based recommender (e.g., factorization machines), the entities take additional arguments:

```python
user = User(0, feature=np.array([0,0,1]))
item = Item(0, feature=np.array([2,1,1]))
event = Event(user, item, context=np.array([0,4,0]))
```

To give an example, a matrix-factorization-based recommender can be used as follows:

```python
import numpy as np
from flurs.recommender.mf import MFRecommender

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
```

## References

FluRS currently supports the following recommendation algorithms:

- Incremental Collaborative Filtering (UserKNN)
  - M. Pepagelis et al. **Incremental Collaborative Filtering for Highly-Scalable Recommendation Algorithms**. In *Foundations of Intelligent Systems*, pp. 553–561, Springer Berlin Heidelberg, 2005.
- Incremental Matrix Factorization (MF)
  - J. Vinagre et al. **[Fast Incremental Matrix Factorization for Recommendation with Positive-only ](http://link.springer.com/chapter/10.1007/978-3-319-08786-3_41)**. In *Proc. of UMAP 2014*, pp. 459–470, July 2014.
- Incremental Matrix Factorization with BPR optimization (BPRMF)
  - S. Rendle et al. **BPR: Bayesian Personalized Ranking from Implicit Feedback**. In *Proc. of UAI 2009*, pp. 452–461, June 2009.
- Incremental Factorization Machines (FM)
  - T. Kitazawa. **[Incremental Factorization Machines for Persistently Cold-Starting Online Item Recommendation](https://arxiv.org/abs/1607.02858)**. arXiv:1607.02858 [cs.LG], July 2016.
- Matrix Sketching (OnlineSketch)
  - T. Kitazawa. **[Sketching Dynamic User-Item Interactions for Online Item Recommendation](http://dl.acm.org/citation.cfm?id=3022152)**. In *Proc. of CHIIR 2017*, March 2017.

Repository [takuti/stream-recommender](https://github.com/takuti/stream-recommender) uses FluRS to write research papers.
