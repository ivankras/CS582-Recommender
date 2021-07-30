import numpy as np
from ast import literal_eval


def missing_counts_by_cols(data):
    missing_val_count_by_column = data.isnull().sum()
    return missing_val_count_by_column[missing_val_count_by_column > 0]


def missing_values_cols(data):
    return [col for col in data.columns if data[col].isnull().any()]


def weighted_rating(data, m, C):
    v = data['vote_count']
    R = data['vote_average']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


def get_content_based_recommendations(title, cosine_sim, indices, data):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def apply_literals(data, features):
    for feature in features:
        data[feature] = data[feature].apply(literal_eval)


def get_list(x):
    """
    Returns the list top 3 elements or entire list; whichever is more.
    :param x:
    :return:
    """
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' '.join(x['director']) + ' ' + ' '.join(x['genres']) #+ ' '.join(x['overview'])
