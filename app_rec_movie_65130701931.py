
import pickle
import streamlit as st
from surprise import SVD


# Load data back from the file
with open('65130701931recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Function to get movie recommendations for a specific user
def get_top_movie_recommendations(user_id, top_n=10):
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    
    # Sort predictions by estimated rating in descending order
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    
    # Get top N movie recommendations
    top_recommendations = sorted_predictions[:top_n]
    
    # Create a list of movie titles and estimated ratings
    recommendations = [(movies[movies['movieId'] == rec.iid]['title'].values[0], rec.est) for rec in top_recommendations]
    return recommendations

# Streamlit app
st.title("Movie Recommendations")

# User input for user ID
user_id = st.number_input("Enter User ID", min_value=1, step=1, value=1)

# Get top 10 movie recommendations
top_recommendations = get_top_movie_recommendations(user_id)

# Display the top 10 recommendations
st.write(f"Top 10 movie recommendations for User {user_id}:")
for movie_title, est_rating in top_recommendations:
    st.write(f"{movie_title} (Estimated Rating: {est_rating:.2f})")

