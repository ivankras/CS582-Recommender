import os
import sys
import json
import os.path
import requests as req
import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader, Dataset, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import (load, dump)
from tensorflow.keras.models import (Model, load_model)
from util import *
from personal.eda import *
from personal.svd import *
from personal.nn import *

movie_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../", "input", "movies_metadata.csv"))
movie_credits_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../", "input", "credits.csv"))
user_ratings_ds = get_ratings_small()

VECTORIZER = None
VECTORIZED_MATRIX = None
COSINE_SIMILARITY_MATRIX = np.array([])
MOVIE_ID_INDICES = None
SVD_MODEL = generate_model_svd()#load(os.path.join(os.path.dirname(__file__), "./personal", "models", "svd"))



def get_trending_movies():
    """ 
    Get popular movies by demographic
    """
    # Clean columns for result
    movies = format_data_objects(movie_df.copy())

    avg_rating = movies['vote_average'].mean()
    min_vote_value = np.percentile(movies['vote_count'], 80)
    movies = movies.loc[movies['vote_count'] >= min_vote_value]
    for index, row in movies.iterrows():
        movies.at[index, 'imdb_rating'] = calculate_weigthed_rating(row['vote_average'], min_vote_value,
                                                                    row['vote_count'], avg_rating)

    top_ten_similar = movies.sort_values('imdb_rating', ascending=False).head(10)
    top_ten_similar = top_ten_similar.apply(get_movie_poster_and_trailer, axis=1)

    return append_imdb_id_to_df(top_ten_similar).to_json(orient='records')

def get_top_10_similar(movie_id, use_overview_for_similarity=False): 
    """
    Given a movie id, return the top 10 similar movies.
    """
    global VECTORIZER
    global VECTORIZED_MATRIX
    global COSINE_SIMILARITY_MATRIX
    global MOVIE_ID_INDICES

    movies_with_credits = movie_credits_df.rename({"movie_id": "id"}, axis='columns').drop('title', axis='columns')
    # create optional variable to use soup or overview. create word soup here.

    movies = movie_df.copy()
    movies = movies.merge(movies_with_credits, on='id')

    # print(movies.isnull().sum())
    for feature in ['overview', 'tagline']:
        movies[feature] = movies[feature].fillna('')

    # Keep calculated objects in memory for performance
    if COSINE_SIMILARITY_MATRIX.size == 0:
        movies = movies.apply(create_movie_column_soup, axis=1)

        VECTORIZER = TfidfVectorizer(stop_words='english')
        if use_overview_for_similarity:
            VECTORIZED_MATRIX = VECTORIZER.fit_transform(movies['overview'])
        else:
            VECTORIZED_MATRIX = VECTORIZER.fit_transform(movies['column_soup'])
        COSINE_SIMILARITY_MATRIX = cosine_similarity(VECTORIZED_MATRIX)
        MOVIE_ID_INDICES = pd.Series(movies.index, index=movies['id']).drop_duplicates()

    movies = format_data_objects(movies)
    # Give recommendation
    movie_similarity_vector = list(enumerate(COSINE_SIMILARITY_MATRIX[MOVIE_ID_INDICES[int(movie_id)]]))
    movie_similarity_scores = sorted(movie_similarity_vector, key=lambda x: x[1], reverse=True)[1:11]
    top_ten_similar = movies.iloc[[i[0] for i in movie_similarity_scores]]
    top_ten_similar = append_imdb_id_to_df(top_ten_similar)
    top_ten_similar = top_ten_similar.apply(get_movie_poster_and_trailer, axis=1)

    return top_ten_similar.to_json(orient='records')


def get_rating(user_id, movie_id):
    """
    Docstring
    This method will compute a predicted rating given a user
    Uses the pre-trained neural network
    """
    prediction = get_top_scores(user_id, 1)
    #_, prediction = predict_score_nn(user_id, movie_id , trained_model=_get_predictor_nn(False, True)[1])
    moviel = []
    moviel.append(int(movie_id))
    movie = format_data_objects(get_movies_from_ids(moviel))
    print(prediction)
    movie['predicted_rating'] = (prediction[0])[1]
    movie = get_movie_poster_and_trailer(movie, True)
    return append_imdb_id_to_df(movie).to_json(orient='records')

def get_rating_svd(user_id, movie_id):
    """
    Docstring
    This method will compute a predicted rating given a user
    Uses the SVD model
    """
    prediction = SVD_MODEL.predict(user_id, movie_id)
    #_, prediction = predict_score_nn(user_id, movie_id , trained_model=_get_predictor_nn(False, True)[1])
    moviel = []
    moviel.append(int(movie_id))
    movie = format_data_objects(get_movies_from_ids(moviel))
    movie['predicted_rating'] = prediction.est
    movie = get_movie_poster_and_trailer(movie, True)
    return append_imdb_id_to_df(movie).to_json(orient='records')

def get_top_10_similar(movie_id, use_overview_for_similarity=False): 
    """
    Given a movie id, return the top 10 similar movies.
    """
    global VECTORIZER
    global VECTORIZED_MATRIX
    global COSINE_SIMILARITY_MATRIX
    global MOVIE_ID_INDICES

    movies_with_credits = movie_credits_df.rename({"movie_id": "id"}, axis='columns')
    # create optional variable to use soup or overview. create word soup here.

    movies = movie_df.copy()
    movies = movies.concat(movies_with_credits, on='id')

    # print(movies.isnull().sum())
    for feature in ['overview', 'tagline']:
        movies[feature] = movies[feature].fillna('')

    # Keep calculated objects in memory for performance
    if COSINE_SIMILARITY_MATRIX.size == 0:
        movies = movies.apply(create_movie_column_soup, axis=1)

        VECTORIZER = TfidfVectorizer(stop_words='english')
        if use_overview_for_similarity:
            VECTORIZED_MATRIX = VECTORIZER.fit_transform(movies['overview'])
        else:
            VECTORIZED_MATRIX = VECTORIZER.fit_transform(movies['column_soup'])
        COSINE_SIMILARITY_MATRIX = cosine_similarity(VECTORIZED_MATRIX)
        MOVIE_ID_INDICES = pd.Series(movies.index, index=movies['id']).drop_duplicates()

    movies = format_data_objects(movies)
    # Give recommendation
    movie_similarity_vector = list(enumerate(COSINE_SIMILARITY_MATRIX[MOVIE_ID_INDICES[int(movie_id)]]))
    movie_similarity_scores = sorted(movie_similarity_vector, key=lambda x: x[1], reverse=True)[1:11]
    top_ten_similar = movies.iloc[[i[0] for i in movie_similarity_scores]]
    movie['imdb_id'] = get_movie_imdb_id(movie['title'].iloc[0])
    movie = get_movie_poster_and_trailer(movie, get_trailer=True)
    return movie.to_json(orient='records')


    
if __name__ == "__main__":
    print(get_rating(12, 862))
    #print(get_trending_movies() + " | " + get_top_10_similar(12) + " | " + get_rating(12, 12))
