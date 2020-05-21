import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import utils as utils

# Read CSV dataset
df = pd.read_csv("movie_dataset.csv")

# Pick features
features = ["keywords", "cast", "genres", "director"]

# Create column that combines all features
for feature in features:
    df[feature] = df[feature].fillna("")

df["combined_features"] = df.apply(utils.combine_features, features=features, axis=1)

# Create count matrix for combined features column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Calculate cosine similarity
cosine_sim = cosine_similarity(count_matrix)
movie_input = "Avatar"

# Get index of movie from title
movie_index = utils.get_index_from_title(movie_input, df)
similar_movies = list(enumerate(cosine_sim[movie_index]))

# Get list of similar movies in descending order
sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

# Print titles of first 50 movies
for i in range(50):
    movie = sorted_movies[i]
    print(utils.get_title_from_index(movie[0], df))
