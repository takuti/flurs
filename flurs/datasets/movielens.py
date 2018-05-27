from ..data.entity import User, Item, Event

import os
import time
import numpy as np
from calendar import monthrange
from datetime import datetime, timedelta

from sklearn.utils import Bunch


def load_movies(data_home, size):
    """Load movie genres as a context.
    Returns:
        dict of movie vectors: item_id -> numpy array (n_genre,)
    """
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

    if size == '100k':
        with open(os.path.join(data_home, 'u.item'), encoding='ISO-8859-1') as f:
            lines = list(map(lambda l: l.rstrip().split('|'), f.readlines()))

        for line in lines:
            movie_vec = np.zeros(n_genre)
            for i, flg_chr in enumerate(line[-n_genre:]):
                if flg_chr == '1':
                    movie_vec[i] = 1.
            movie_id = int(line[0])
            movies[movie_id] = movie_vec
    elif size == '1m':
        with open(os.path.join(data_home, 'movies.dat'), encoding='ISO-8859-1') as f:
            lines = list(map(lambda l: l.rstrip().split('::'), f.readlines()))

        for item_id_str, title, genres in lines:
            movie_vec = np.zeros(n_genre)
            for genre in genres.split('|'):
                i = all_genres.index(genre)
                movie_vec[i] = 1.
            item_id = int(item_id_str)
            movies[item_id] = movie_vec

    return movies


def load_users(data_home, size):
    """Load user demographics as contexts.User ID -> {sex (M/F), age (7 groupd), occupation(0-20; 21)}
    Returns:
        dict of user vectors: user_id -> numpy array (1+1+21,); (sex_flg + age_group + n_occupation, )
    """
    ages = [1, 18, 25, 35, 45, 50, 56, 999]

    users = {}

    if size == '100k':
        all_occupations = ['administrator',
                           'artist',
                           'doctor',
                           'educator',
                           'engineer',
                           'entertainment',
                           'executive',
                           'healthcare',
                           'homemaker',
                           'lawyer',
                           'librarian',
                           'marketing',
                           'none',
                           'other',
                           'programmer',
                           'retired',
                           'salesman',
                           'scientist',
                           'student',
                           'technician',
                           'writer']

        with open(os.path.join(data_home, 'u.user'), encoding='ISO-8859-1') as f:
            lines = list(map(lambda l: l.rstrip().split('|'), f.readlines()))

        for user_id_str, age_str, sex_str, occupation_str, zip_code in lines:
                user_vec = np.zeros(1 + 1 + 21)  # 1 categorical, 1 value, 21 categorical
                user_vec[0] = 0 if sex_str == 'M' else 1  # sex

                # age (ML1M is "age group", but 100k has actual "age")
                age = int(age_str)
                for i in range(7):
                    if age >= ages[i] and age < ages[i + 1]:
                        user_vec[1] = i
                        break

                user_vec[2 + all_occupations.index(occupation_str)] = 1  # occupation (1-of-21)
                users[int(user_id_str)] = user_vec
    elif size == '1m':
        with open(os.path.join(data_home, 'users.dat'), encoding='ISO-8859-1') as f:
            lines = list(map(lambda l: l.rstrip().split('::'), f.readlines()))

        for user_id_str, sex_str, age_str, occupation_str, zip_code in lines:
                user_vec = np.zeros(1 + 1 + 21)  # 1 categorical, 1 value, 21 categorical
                user_vec[0] = 0 if sex_str == 'M' else 1  # sex
                user_vec[1] = ages.index(int(age_str))  # age group (1, 18, ...)
                user_vec[2 + int(occupation_str)] = 1  # occupation (1-of-21)
                users[int(user_id_str)] = user_vec

    return users


def load_ratings(data_home, size):
    """Load all samples in the dataset.
    """

    if size == '100k':
        with open(os.path.join(data_home, 'u.data'), encoding='ISO-8859-1') as f:
            lines = list(map(lambda l: list(map(int, l.rstrip().split('\t'))), f.readlines()))
    elif size == '1m':
        with open(os.path.join(data_home, 'ratings.dat'), encoding='ISO-8859-1') as f:
            lines = list(map(lambda l: list(map(int, l.rstrip().split('::'))), f.readlines()))

    ratings = []

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


def fetch_movielens(data_home=None, size='100k'):
    assert data_home is not None

    if size not in ('100k', '1m'):
        raise ValueError("size can only be '100k' or '1m', got %s" % size)

    ratings = load_ratings(data_home, size)
    users = load_users(data_home, size)
    movies = load_movies(data_home, size)

    samples = []

    user_ids = {}
    item_ids = {}

    head_date = datetime(*time.localtime(ratings[0, 3])[:6])
    dts = []

    last = {}

    for user_id, item_id, rating, timestamp in ratings:
        # give an unique user index
        if user_id in user_ids:
            u_index = user_ids[user_id]
        else:
            u_index = len(user_ids)
            user_ids[user_id] = u_index

        # give an unique item index
        if item_id in item_ids:
            i_index = item_ids[item_id]
        else:
            i_index = len(item_ids)
            item_ids[item_id] = i_index

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

    # contexts in this dataset
    # 1 delta time, 18 genres, and 23 demographics (1 for M/F, 1 for age, 21 for occupation(0-20))
    # 7 for day of week, 18 for the last rated item genres, 7 for the last day of week
    return Bunch(samples=samples,
                 can_repeat=False,
                 contexts={'others': 7 + 18 + 7, 'item': 18, 'user': 23},
                 n_user=len(user_ids),
                 n_item=len(item_ids),
                 n_sample=len(samples))
