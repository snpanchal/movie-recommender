# Utils file for movie recommender


def combine_features(row, features):
    combined = ""
    for feature in features:
        combined += (row[f] + " ")
    return combined
