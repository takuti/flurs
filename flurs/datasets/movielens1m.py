from ..data.entity import User, Item, Event

import os
import time
import numpy as np
from calendar import monthrange
from datetime import datetime, timedelta

from sklearn.utils import Bunch


def load_movies(movies_path):
    """Load movie genres as a context.
    Returns:
        dict of movie vectors: item_id -> numpy array (n_genre,)
    """
    with open(movies_path, encoding='ISO-8859-1') as f:
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


def load_users(users_path):
    """Load user demographics as contexts.User ID -> {sex (M/F), age (7 groupd), occupation(0-20; 21)}
    Returns:
        dict of user vectors: user_id -> numpy array (1+1+21,); (sex_flg + age_group + n_occupation, )
    """
    with open(users_path, encoding='ISO-8859-1') as f:
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


def load_ratings(ratings_path):
    """Load all samples in the dataset.
    """
    ratings = []
    with open(ratings_path, encoding='ISO-8859-1') as f:
        lines = list(map(lambda l: list(map(int, l.rstrip().split('::'))), f.readlines()))
        for l in lines:
            # Since we consider positive-only feedback setting, ratings < 5 will be excluded.
            if l[2] == 5:
                ratings.append(l)
    ratings = np.asarray(ratings)

    # sorted by timestamp
    return ratings[np.argsort(ratings[:, 3])]


def delta(d1, d2, opt='d'):
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


def fetch_movielens1m(data_home):
    samples = []

    ratings = load_ratings(os.path.join(data_home, 'ratings.dat'))
    users = load_users(os.path.join(data_home, 'users.dat'))
    movies, movie_titles = load_movies(os.path.join(data_home, 'movies.dat'))

    user_ids = []
    item_ids = []

    head_date = datetime(*time.localtime(ratings[0, 3])[:6])
    dts = []

    last = {}

    for user_id, item_id, rating, timestamp in ratings:
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
        dt = delta(head_date, date)
        dts.append(dt)

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
        samples.append(sample)

        # record users' last rated movie features
        last[user_id] = {'item': movies[item_id], 'weekday': weekday_vec}

    n_sample = len(samples)
    n_batch_train = int(n_sample * 0.2)  # 20% for pre-training to avoid cold-start
    n_batch_test = int(n_sample * 0.1)  # 10% for evaluation of pre-training

    # contexts in this dataset
    # 1 delta time, 18 genres, and 23 demographics (1 for M/F, 1 for age, 21 for occupation(0-20))
    # 7 for day of week, 18 for the last rated item genres, 7 for the last day of week
    return Bunch(samples=samples,
                 can_repeat=False,
                 contexts={'others': 7 + 18 + 7, 'item': 18, 'user': 23},
                 n_user=len(user_ids),
                 n_item=len(item_ids),
                 n_sample=n_sample,
                 n_batch_train=n_batch_train,
                 n_batch_test=n_batch_test,
                 n_test=n_sample - (n_batch_train + n_batch_test))
