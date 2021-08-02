# EDA

import pandas as pd
from utils import missing_counts_by_cols
import numpy as np

tmdb_movies_data = pd.read_csv("/Users/umairsaeed/Desktop/dataset/movies.csv")
tmdb_credits_data = pd.read_csv("/Users/umairsaeed/Desktop/dataset/credits.csv")


tmdb_credits_data.columns = ['id', 'tittle', 'cast', 'crew']
tmdb_merged_data = tmdb_movies_data.merge(tmdb_credits_data, on="id")

tmdb_merged_data.head()



missing_counts_by_cols(tmdb_merged_data)

def getMergedData():
    return tmdb_merged_data
