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
from ast import literal_eval


def get_movie_imdb_id(title, get_object=False):
    """
    Get the movie's imdb id using OMDB API
    """
    if "OMDB_API_KEY" in os.environ:
        OMDB_KEY = os.environ['OMDB_API_KEY']
    else:
        error_message = "ERROR: OMDB_API_KEY environment variable must be set, run 'source OMDB_API_KEY'" \
                        " after putting you key in it on project root directory"
        print(error_message, file=sys.stderr)
        return error_message, 500

    url = "http://www.omdbapi.com/?t=%s&apikey=%s" % (title, OMDB_KEY)
    res = json.loads(req.get(url).text)

    if get_object:
        return res

    imdb_id = res["imdbID"] if "imdbID" in res else "not_found"
    return imdb_id

def get_movie_id_by_title(title):
    """
    Get the id for a movie title
    """
    return str(movie_df.loc[movie_df['title'] == title]['id'].iloc[0])


def get_movie_poster_and_trailer(movie, get_trailer=False):
    """ Get movie and append poster image from TMDB API"""
    if "TMDB_API_KEY" in os.environ:
        TMDB_KEY = os.environ['TMDB_API_KEY']
    else:
        error_message = "Error TMDB API Key not set. Refer to documentation to add this"
        print(error_message, file=sys.stderr)
        return error_message, 500
    movie_id = movie['id']
    if not isinstance(movie_id, int):
        movie_id = movie['id'].iloc[0]

    # The URL for the poster has two parts, base_url+poster_size and actual URL. The first is found in config.
    tmdb_config_url = "https://api.themoviedb.org/3/configuration?api_key=%s" % TMDB_KEY
    url = "https://api.themoviedb.org/3/movie/%s?api_key=%s&append_to_response=videos" % (str(movie_id), TMDB_KEY)

    config_res = json.loads(req.get(tmdb_config_url).text)["images"]
    res = json.loads(req.get(url).text)

    base_url = config_res["base_url"]
    poster_size = config_res["poster_sizes"][3]
    if res["poster_path"]:
        poster_url = res["poster_path"]
        movie['poster_url'] = base_url + poster_size + poster_url
    else:
        movie['poster_url'] = "https://www.publicdomainpictures.net/pictures/280000/velka/not-found-image-15383864787lu.jpg"

    if get_trailer:
        trailer_url_id = {}
        if res["videos"]["results"]:
            trailer_url_id = next(video for video in res["videos"]["results"] if video["type"] == "Trailer" and "teaser" not in video["name"].lower())
            if not trailer_url_id:
                trailer_url_id = res["videos"]["results"][0]

        if trailer_url_id and trailer_url_id["site"] == "YouTube":
            #movie["trailer_url"] = "https://www.youtube.com/watch?v=%s" % trailer_url_id["key"]
            movie["trailer_url"] = "https://www.youtube.com/embed/%s?autoplay=1" % trailer_url_id["key"]
        else:
            movie["trailer_url"] = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" #not found

    return movie

def format_data_objects(dataframe):
    dataframe = dataframe.drop([
        'homepage', 'keywords', 'original_language', 'production_countries', 'original_title', 'revenue',
        'spoken_languages', 'status', 'production_companies', 'crew', 'cast', 'column_soup'
    ], axis='columns', errors='ignore')
    dataframe['genres'] = dataframe['genres'].apply(literal_eval)
    return dataframe

def append_imdb_id_to_df(dataframe):
    for index, row in dataframe.iterrows():
        dataframe.at[index, 'imdb_id'] = get_movie_imdb_id(row['title'])
    return dataframe


def calculate_weigthed_rating(rating, minimum_votes, number_of_votes, avg_rating):
    """ 
    Calculate weigted rating as calculated in IMDB. 
    This rating accounts for the number of votes a movie has.
    """
    rhs = number_of_votes / (number_of_votes + minimum_votes)
    lhs = minimum_votes / (number_of_votes + minimum_votes)
    return (rhs * rating) + (lhs * avg_rating)


def create_movie_column_soup(movie):
    """ 
    Given a group of dataframes and a group of labels,
    check if the label is in each df then add it's values to the soup if they are strings.
    """
    # serialize important cast names
    desired_crew_jobs = ['Original Music Composer', 'Director', 'Writer' ]
    genres = stringify_features(movie, 'genres').lower()
    keywords = stringify_features(movie, 'keywords').lower()
    overview = movie['overview'].lower()
    cast = ' '.join(['-'.join(i['name'].split(" ")) for i in sorted(literal_eval(movie['cast']), key=lambda x: x['order'])[0:10]]).lower()
    crew = ' '.join(['-'.join(i['name'].split(" ")) for i in literal_eval(movie['crew']) if i['job'] in desired_crew_jobs]).lower()
    # serialize important crew names departments:

    movie['column_soup'] = "%s %s %s %s %s" % (genres, keywords, overview, cast, crew)
    return movie


def stringify_features(items, feature, extract_feature='name'):
    return ' '.join(['-'.join(i[extract_feature].split(" ")) for i in literal_eval(items[feature])])