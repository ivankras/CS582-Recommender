from trending.utils import *

C = tmdb_movies_data['vote_average'].mean()
m = tmdb_movies_data['vote_count'].quantile(0.9)

q_movies = tmdb_movies_data.copy().loc[tmdb_movies_data['vote_count'] >= m]

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(lambda movie: weighted_rating(data=movie, m=m, C=C), axis=1)

# Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)


def get_top_k_movies(k=10):
    return q_movies[:k]


if __name__ == '__main__':
    print(get_top_k_movies(10))
