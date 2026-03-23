import streamlit as st
from recommender import get_recommendation

st.title("🎬 Movie Recommender System")

movie_name = st.text_input("Enter Movie Name")

if st.button("Recommend"):

    if movie_name == "":
        st.warning("Enter a movie name")

    else:
        results = get_recommendation(movie_name)

        if results is None:
            st.error("Movie not found!")
        else:
            for i, movie in enumerate(results, 1):
                st.write(f"{i}. {movie}")