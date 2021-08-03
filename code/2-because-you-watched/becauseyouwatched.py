from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')
# Replace NaN with an empty string
tmdb_merged_data['overview'] = tmdb_merged_data['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(tmdb_merged_data['overview'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(tmdb_merged_data.index, index=tmdb_merged_data['title']).drop_duplicates()

get_content_based_recommendations(title='The Dark Knight Rises',
                                  cosine_sim=cosine_sim,
                                  data=tmdb_merged_data,
                                  indices=indices)


def get_k_recommendations_based_on_overview(title, k=10):
    return get_content_based_recommendations(title=title,
                                             cosine_sim=cosine_sim,
                                             data=tmdb_merged_data,
                                             indices=indices)[:k]


features = ['cast', 'crew', 'keywords', 'genres']

apply_literals(data=tmdb_merged_data, features=features)

# Define new director, cast, genres and keywords features that are in a suitable form.
tmdb_merged_data['director'] = tmdb_merged_data['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    tmdb_merged_data[feature] = tmdb_merged_data[feature].apply(get_list)

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    tmdb_merged_data[feature] = tmdb_merged_data[feature].apply(clean_data)

tmdb_merged_data['soup'] = tmdb_merged_data.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(tmdb_merged_data['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our main DataFrame and construct reverse mapping as before
tmdb_merged_data = tmdb_merged_data.reset_index()
indices = pd.Series(tmdb_merged_data.index, index=tmdb_merged_data['title'])

get_content_based_recommendations(title='The Dark Knight Rises',
                                  cosine_sim=cosine_sim2,
                                  indices=indices,
                                  data=tmdb_merged_data)


def get_k_recommendations_based_on_soup(title, k=10):
    return get_content_based_recommendations(title=title,
                                             cosine_sim=cosine_sim2,
                                             indices=indices,
                                             data=tmdb_merged_data)[:k]


if __name__ == '__main__':
    print(get_k_recommendations_based_on_soup(title='The Dark Knight Rises', k=10))
