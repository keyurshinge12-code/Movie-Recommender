# Movie Recommender System

## Live Demo
https://movie-recommender-amkaqehqk8fouwpx4tdrln.streamlit.app/


# Project Overview

This project is a "Movie Recommendation System" that suggests similar movies based on user preferences.
It uses "Collaborative Filtering" to find patterns in user ratings and recommend movies that users with similar tastes have liked.

The goal of this project was not just to build a model, but to understand how recommendation systems work in real-world applications and deploy it as an interactive web app.

---

#How It Works

Created a "User-Movie Matrix" using ratings data
Handled missing values by filling with zeros
Reduced noise by:
1 ->  Removing movies with very few ratings
2 ->  Removing inactive users

Converted the data into a "Sparse Matrix" for efficiency
Used "K-Nearest Neighbors (KNN)" with cosine similarity
Found similar movies based on distance between vectors


# Tech Stack

-> Python
-> Pandas & NumPy
-> Scikit-learn
-> SciPy
-> Streamlit

# Project Structure

movie-recommender/
│
├── app.py # Streamlit app (UI)
├── recommender.py # Recommendation logic
├── train_model.py # Model training script
├── requirements.txt #Dependencies
├── README.md # Project Documentation
│
├── data/
│ ├── movies.csv # Movie dataset
│ ├── ratings.csv # User ratings dataset
│
├── models/
│ ├── movies.pkl # Saved movies data
│ ├── final_data.pkl # Processed user-movie matrix
│ ├── knn_model.pkl # Trained KNN model
│ ├── csr_data.pkl # Sparse matrix


# How to Run Locally

1 -> Clone The Repository

git clone https://github.com/your-username/movie-recommender.git

2 -> Navigate to the project folder

cd movie-recommender

3 -> Create virtual environment

python -m venv venv
venv\Scripts\activate

4 -> Install dependencies

pip install -r requirements.txt OR 
You can just download the needed libraries which are: 
streamlit, numpy, scikit learn, scipy, pandas

5 -> Run the app

streamlit run app.py


# Features

-> Search for any movie
-> Get top similar movie recommendations
-> Fast and efficient using sparse matrix
-> Simple and interactive UI


# What I Learned

-> How recommendation systems work using collaborative filtering
-> Handling sparse datasets efficiently
-> Improving model quality by reducing noise
-> Deploying ML models using Streamlit
-> Structuring real-world ML projects


# Future Improvements

-> Add movie posters using API
-> Improve recommendation quality
-> Add search suggestions/autocomplete
-> Deploy with better UI/UX


# Final Thoughts

This project helped me bridge the gap between "Learning Machine Learning" and "Building Real-World Applications".
It was a great experience working through data preprocessing, model building, and deployment.

