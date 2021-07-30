import pandas as pd
from utils import *

tmdb_movies_data = pd.read_csv("../../dataset/TMDB 5000 Movie Dataset/tmdb_5000_movies.csv")
tmdb_credits_data = pd.read_csv("../../dataset/TMDB 5000 Movie Dataset/tmdb_5000_credits.csv")

C = tmdb_movies_data['vote_average'].mean()
m = tmdb_movies_data['vote_count'].quantile(0.9)

q_movies = tmdb_movies_data.copy().loc[tmdb_movies_data['vote_count'] >= m]

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(lambda movie: weighted_rating(data=movie, m=m, C=C), axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

from flask import Flask

app = Flask(__name__)


@app.route("/demographic")
def demographic():
    return q_movies


if __name__ == '__main__':
    app.run()

