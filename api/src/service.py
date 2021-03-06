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
from trending.demographic import *
from history.becauseyouwatched import *

movie_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../", "input", "movies_metadata.csv"))
movie_credits_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../", "input", "credits.csv"))
user_ratings_ds = get_ratings_small()

SVD_MODEL = generate_model_svd()



def get_trending_movies():
    """ 
    Get popular movies by demographic
    """
    # Clean columns for result
    movies = get_top_k_movies(10)
    movies = append_imdb_id_to_df(movies)
    movies = format_data_objects(movies)
    movies = movies.apply(get_movie_poster_and_trailer, axis=1, get_trailer=True)
    
    return movies.to_json(orient='records')

def get_top_10_similar(movie_id, use_overview_for_similarity=False): 
    """
    Given a movie id, return the top 10 similar movies.
    """
    movie = []
    movie.append(int(movie_id))
    movie = get_movies_from_ids(movie)
    title = str(movie['title'].item())
    
    top_ten_similar = get_k_recommendations_based_on_soup(title , 10).tolist()
    ds = get_movies_dataset()
    top_ten_ids = [int(get_movie_id_by_title(ds, str(x))) for x in top_ten_similar]
    top_ten_similar = format_data_objects(get_movies_from_ids(top_ten_ids))
    top_ten_similar = top_ten_similar.apply(get_movie_poster_and_trailer, axis=1, get_trailer=True)
    top_ten_similar = append_imdb_id_to_df(top_ten_similar)

    return top_ten_similar.to_json(orient='records')


def get_rating(user_id):
    """
    Get top 10 most highly rated movies based on user's watch history
    Uses the pre-trained neural network
    """
    prediction = get_top_scores(user_id, 10)
    #_, prediction = predict_score_nn(user_id, movie_id , trained_model=_get_predictor_nn(False, True)[1])
    moviel = [x[0] for x in prediction]
    movie = format_data_objects(get_movies_from_ids(moviel))
    #movie['predicted_rating'] = (prediction[0])[1]
    movies = movie.apply(get_movie_poster_and_trailer, axis=1, get_trailer=True)
    return append_imdb_id_to_df(movies).to_json(orient='records')

def get_rating_svd(user_id):
    """
    Get top 10 most highly rated movies based on user's watch history
    Uses the SVD model
    """
    prediction = get_top_k_predictions(get_all_predictions_svd(user_id), 10)
    print(prediction[0][0])
    moviel = [x[1] for x in prediction]
    movie = format_data_objects(get_movies_from_ids(moviel))
    movies = movie.apply(get_movie_poster_and_trailer, axis=1, get_trailer=True)
    return append_imdb_id_to_df(movies).to_json(orient='records')

def get_movie_by_id(movie_id):
    mids = []
    mids.append(int(movie_id))
    movie = format_data_objects(get_movies_from_ids(mids))
    movie = get_movie_poster_and_trailer(movie, True)
    return append_imdb_id_to_df(movie).to_json(orient='records')
    
if __name__ == "__main__":
    print(get_rating(12, 862))