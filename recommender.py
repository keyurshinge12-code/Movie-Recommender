import pickle
import pandas as pd

movies = pickle.load(open("models/movies.pkl", "rb"))
final_data = pickle.load(open("models/final_data.pkl", "rb"))
knn = pickle.load(open("models/knn_model.pkl", "rb"))
csr_data = pickle.load(open("models/csr_data.pkl", "rb"))

def get_recommendation(movie_name, n_recommendation=10):


    # Find The Matching Movie
    movie_list = movies[movies["title"].str.contains(movie_name, case=False, na=False)]

    if movie_list.empty:
        return None

    # Get The First Match For Movie Id
    movie_id = movie_list.iloc[0]["movieId"]

    if movie_id not in final_data["movieId"].values:
        return None

     # Get The Correct Movie Index
    movie_idx = final_data[final_data["movieId"] == movie_id].index[0]

    # Find Nearest Neighbors
    distances, indices = knn.kneighbors(
        csr_data[movie_idx].reshape(1, -1),
        n_neighbors=n_recommendation + 1
    )

    # Get The Result List
    rec_movies = []

    for i in indices.flatten()[1:]:
        rec_movie_id = final_data.iloc[i]["movieId"]
        title = movies[movies["movieId"] == rec_movie_id]["title"].values[0]
        rec_movies.append(title)

    return rec_movies