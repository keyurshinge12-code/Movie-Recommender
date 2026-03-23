import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
import os

# Create Models Folder
os.makedirs("models", exist_ok=True)

# Load Data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Pivot Table
final_data = ratings.pivot(index="movieId", columns="userId", values="rating")
final_data.fillna(0, inplace=True)

# Filtering
no_user_voted = ratings.groupby("movieId")['rating'].count()
no_movies_voted = ratings.groupby("userId")['rating'].count()

final_data = final_data.loc[no_user_voted[no_user_voted > 10].index, :]
final_data = final_data.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Reset Index 
final_data.reset_index(inplace=True)

# Create Sparse Matrix
csr_data = csr_matrix(final_data.drop("movieId", axis=1).values)

# Train Model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Save Everything
pickle.dump(movies, open("models/movies.pkl", "wb"))
pickle.dump(final_data, open("models/final_data.pkl", "wb"))
pickle.dump(knn, open("models/knn_model.pkl", "wb"))
pickle.dump(csr_data, open("models/csr_data.pkl", "wb"))

print("Model Training Complete And Saved")