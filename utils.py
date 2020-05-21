# Utils file for movie recommender


def combine_features(row, features):
    combined = ""
    for feature in features:
        combined += (row[feature] + " ")
    return combined


def get_title_from_index(index, df):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title, df):
    return df[df.title == title]["index"].values[0]
