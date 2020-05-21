import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import combine_features

# Read CSV dataset
df = pd.read_csv("movie_dataset.csv")

# Pick features
features = ["keywords", "cast", "genres", "director"]

# Create column that combines all features
for feature in features:
    df[feature] = df[feature].fillna("")

df["combined_features"] = df.apply(combine_features, axis=1)

# Create count matrix for combined features column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Calculate cosine similarity
cosine_sim = cosine_similarity(count_matrix)
