"""USAGE: %(program)s PATH_TO_MOVIELENS_1M_DIR
"""

from flurs.data.entity import User, Item, Event
from flurs.recommender import FMRecommender
from flurs.evaluator import Evaluator

import logging
import numpy as np
import time
import os
import sys
from calendar import monthrange
from datetime import datetime, timedelta


class MovieLens1MConverter:

    def __init__(self, path_to_dir):
        self.path = {
            'ratings': os.path.join(path_to_dir, 'ratings.dat'),
            'items': os.path.join(path_to_dir, 'movies.dat'),
            'users': os.path.join(path_to_dir, 'users.dat')
        }

        # contexts in this dataset
        # 1 delta time, 18 genres, and 23 demographics (1 for M/F, 1 for age, 21 for occupation(0-20))
        # 7 for day of week, 18 for the last rated item genres, 7 for the last day of week
        self.contexts = {'others': 7 + 18 + 7, 'item': 18, 'user': 23}

        self.can_repeat = False

    def convert(self):
        """Create a list of samples and count number of users/items.
        """
        self.__load_ratings()

        users = self.__load_users()
        movies, movie_titles = self.__load_movies()

        user_ids = []
        item_ids = []

        self.samples = []

        head_date = datetime(*time.localtime(self.ratings[0, 3])[:6])
        self.dts = []

        last = {}

        for user_id, item_id, rating, timestamp in self.ratings:
            # give an unique user index
            if user_id not in user_ids:
                user_ids.append(user_id)
            u_index = user_ids.index(user_id)

            # give an unique item index
            if item_id not in item_ids:
                item_ids.append(item_id)
            i_index = item_ids.index(item_id)

            # delta days
            date = datetime(*time.localtime(timestamp)[:6])
            dt = self.__delta(head_date, date)
            self.dts.append(dt)

            weekday_vec = np.zeros(7)
            weekday_vec[date.weekday()] = 1

            if user_id in last:
                last_item_vec = last[user_id]['item']
                last_weekday_vec = last[user_id]['weekday']
            else:
                last_item_vec = np.zeros(18)
                last_weekday_vec = np.zeros(7)

            others = np.concatenate((weekday_vec, last_item_vec, last_weekday_vec))

            user = User(u_index, users[user_id])
            item = Item(i_index, movies[item_id])

            sample = Event(user, item, 1., others)
            self.samples.append(sample)

            # record users' last rated movie features
            last[user_id] = {'item': movies[item_id], 'weekday': weekday_vec}

        self.n_user = len(user_ids)
        self.n_item = len(item_ids)
        self.n_sample = len(self.samples)
        self.n_batch_train = int(self.n_sample * 0.2)  # 20% for pre-training to avoid cold-start
        self.n_batch_test = int(self.n_sample * 0.1)  # 10% for evaluation of pre-training
        self.n_test = self.n_sample - (self.n_batch_train + self.n_batch_test)

    def __load_movies(self):
        """Load movie genres as a context.
        Returns:
            dict of movie vectors: item_id -> numpy array (n_genre,)
        """
        with open(self.path['items'], encoding='ISO-8859-1') as f:
            lines = list(map(lambda l: l.rstrip().split('::'), f.readlines()))

        all_genres = ['Action',
                      'Adventure',
                      'Animation',
                      "Children's",
                      'Comedy',
                      'Crime',
                      'Documentary',
                      'Drama',
                      'Fantasy',
                      'Film-Noir',
                      'Horror',
                      'Musical',
                      'Mystery',
                      'Romance',
                      'Sci-Fi',
                      'Thriller',
                      'War',
                      'Western']
        n_genre = len(all_genres)

        movies = {}
        movie_titles = {}
        for item_id_str, title, genres in lines:
            movie_vec = np.zeros(n_genre)
            for genre in genres.split('|'):
                i = all_genres.index(genre)
                movie_vec[i] = 1.
            item_id = int(item_id_str)
            movies[item_id] = movie_vec
            movie_titles[item_id] = title

        return movies, movie_titles

    def __load_users(self):
        """Load user demographics as contexts.User ID -> {sex (M/F), age (7 groupd), occupation(0-20; 21)}
        Returns:
            dict of user vectors: user_id -> numpy array (1+1+21,); (sex_flg + age_group + n_occupation, )
        """
        with open(self.path['users'], encoding='ISO-8859-1') as f:
            lines = list(map(lambda l: l.rstrip().split('::'), f.readlines()))

        ages = [1, 18, 25, 35, 45, 50, 56]

        users = {}
        for user_id_str, sex_str, age_str, occupation_str, zip_code in lines:
                user_vec = np.zeros(1 + 1 + 21)  # 1 categorical, 1 value, 21 categorical
                user_vec[0] = 0 if sex_str == 'M' else 1  # sex
                user_vec[1] = ages.index(int(age_str))  # age group (1, 18, ...)
                user_vec[2 + int(occupation_str)] = 1  # occupation (1-of-21)
                users[int(user_id_str)] = user_vec

        return users

    def __load_ratings(self):
        """Load all samples in the dataset.
        """
        ratings = []
        with open(self.path['ratings'], encoding='ISO-8859-1') as f:
            lines = list(map(lambda l: list(map(int, l.rstrip().split('::'))), f.readlines()))
            for l in lines:
                # Since we consider positive-only feedback setting, ratings < 5 will be excluded.
                if l[2] == 5:
                    ratings.append(l)
        self.ratings = np.asarray(ratings)

        # sorted by timestamp
        self.ratings = self.ratings[np.argsort(self.ratings[:, 3])]

    def __delta(self, d1, d2, opt='d'):
        """Compute difference between given 2 dates in month/day.
        """
        delta = 0

        if opt == 'm':
            while True:
                mdays = monthrange(d1.year, d1.month)[1]
                d1 += timedelta(days=mdays)
                if d1 <= d2:
                    delta += 1
                else:
                    break
        else:
            delta = (d2 - d1).days

        return delta


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))

    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    logging.info('converting data into FluRS input object')
    data = MovieLens1MConverter(sys.argv[1])
    data.convert()

    logging.info('initialize recommendation model and evaluation module')
    rec = FMRecommender(p=sum(data.contexts.values()),  # number of dimensions of input vector
                        k=40,
                        l2_reg_w0=2.,
                        l2_reg_w=8.,
                        l2_reg_V=16.,
                        learn_rate=.004)
    rec.initialize()
    evaluator = Evaluator(rec, data.can_repeat)

    batch_tail = data.n_batch_train + data.n_batch_test

    # pre-train
    # 20% for batch training | 10% for batch evaluate
    # after the batch training, 10% samples are used for incremental updating
    logging.info('batch pre-training before streaming input')
    evaluator.fit(
        data.samples[:data.n_batch_train],
        data.samples[data.n_batch_train:batch_tail],
        n_epoch=1  # single pass even for batch training
    )

    # 70% incremental evaluation and updating
    logging.info('incrementally predict, evaluate and update the recommender')
    res = evaluator.evaluate(data.samples[batch_tail:])

    print(res)
