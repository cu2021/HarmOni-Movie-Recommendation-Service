import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
from surprise import PredictionImpossible
import os
from dotenv import load_dotenv
import pickle

# Load environment variables from .env file
load_dotenv()


def load_data():
    """
    Load datasets for MovieLens, TMDb links, and movie metadata.

    Returns:
    tuple: DataFrames containing ratings, links, and merged movie details.
    """
    ratings_df = pd.read_csv(r"data/ratings_small.csv")  # MovieLens ratings
    links_df = pd.read_csv(r"data/links_small.csv")  # TMDb and MovieLens ID mapping
    new_df = pd.read_csv(r"data/new_df_data (1).csv")  # Movies Dataset

    return ratings_df, links_df, new_df


# def load_similarity_matrix():
#     """
#     Load the precomputed content-based cosine similarity matrix.

#     Returns:
#     numpy.ndarray: Cosine similarity matrix.
#     """
#     return np.load(r"data/cosine_similarity2.npy")


def load_model():
    """
    Load pre-trained SVD model for collaborative filtering.

    Returns:
    object: Pre-trained SVD model.
    """
    with open(r"models/best_svd1.pkl", "rb") as file:
        return pickle.load(file)


# Load Data & Model at Startup
ratings_df, links_df, new_df = load_data()
best_svd1 = load_model()

# Load Count Matrix
count_matrix = load_npz(r"data/count_matrix.npz")

# Create Movie Index Mapping (Title to Index)
indices = pd.Series(new_df.index, index=new_df["title"]).drop_duplicates()


def weighted_rating(x, C, m):
    """
    Computes IMDb-style weighted rating.

    Parameters:
    x (Series): A row of a DataFrame containing vote count and vote average.
    C (float): Mean vote across all movies.
    m (float): Minimum votes required to be considered.

    Returns:
    float: IMDb-weighted rating.
    """
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (v + m) * C)


def match_tmdb_to_movielens(qualified_movies, links_df):
    """
    Match TMDb ID with MovieLens movieId before making SVD predictions.

    Parameters:
    qualified_movies (DataFrame): Movies selected for recommendation.
    links_df (DataFrame): Mapping of TMDb and MovieLens IDs.

    Returns:
    DataFrame: Updated movies DataFrame with matched MovieLens IDs.
    """
    qualified_movies = qualified_movies.merge(
        links_df[["tmdbId", "movieId"]], left_on="id", right_on="tmdbId", how="left"
    )
    return qualified_movies.dropna(subset=["movieId"]).astype({"movieId": "int"})


def handle_new_user(
    qualified_movies, top_n, popularity_weight, similarity_weight, recency_weight
):
    """
    Return recommendations for new users based only on content similarity and IMDb scores.

    Parameters:
    qualified_movies (DataFrame): Movies selected for recommendation.
    top_n (int): Number of recommendations.
    popularity_weight (float): Weight of IMDb rating.
    similarity_weight (float): Weight of content similarity.

    Returns:
    DataFrame: Top N recommendations for a new user.
    """
    # Compute Final Score for Sorting, Final Score (no SVD for new user)
    qualified_movies["final_score"] = (
        popularity_weight * qualified_movies["weighted_rating"]
    ) + (similarity_weight * qualified_movies["similarity_score"])

    # Combine recency into final_score
    qualified_movies["final_score"] = (
        qualified_movies["final_score"] * (1 - recency_weight)
        + qualified_movies["recency_score"] * recency_weight
    )

    return qualified_movies.sort_values("final_score", ascending=False).head(top_n)[
        [
            "id",
            "title",
            "release_date",
            "final_score",
        ]
    ]


def predict_ratings(user_id, qualified_movies, best_svd_model, ratings_df, C):
    """
    Predict ratings using the trained SVD model or fallback to nearest neighbor ratings.

    Parameters:
    user_id (int): User ID for whom ratings are predicted.
    qualified_movies (DataFrame): Movies selected for recommendation.
    best_svd_model (object): Trained SVD model.
    ratings_df (DataFrame): User ratings dataset.
    C (float): Mean vote across all movies.

    Returns:
    DataFrame: Updated movies DataFrame with predicted ratings.
    """
    predicted_ratings = []

    for movie_id in qualified_movies["movieId"]:
        try:
            prediction = best_svd_model.predict(user_id, movie_id)
            predicted_ratings.append(prediction.est)  # Extract predicted rating
        except PredictionImpossible:
            # Use weighted average of nearest neighbors if prediction fails
            nearest_neighbors = ratings_df[ratings_df["movieId"] == movie_id]["rating"]
            predicted_ratings.append(
                nearest_neighbors.mean() if not nearest_neighbors.empty else C
            )

    qualified_movies["predicted_rating"] = predicted_ratings
    return qualified_movies


def compute_hybrid_score(qualified_movies, user_ratings_count, recency_weight, top_n):
    """
    Compute final hybrid recommendation score with dynamic weighting.

    Parameters:
    qualified_movies (DataFrame): Movies selected for recommendation.
    user_ratings_count (int): Number of ratings given by the user.

    Returns:
    DataFrame: Movies DataFrame with hybrid scores.
    """
    # Dynamic Weighting
    if user_ratings_count < 10:
        svd_weight = 0.5
    elif user_ratings_count < 50:
        svd_weight = 0.6
    else:
        svd_weight = 0.7

    imdb_weight = 1 - svd_weight
    qualified_movies["final_score"] = (
        svd_weight * qualified_movies["predicted_rating"]
        + imdb_weight * qualified_movies["weighted_rating"]
    )

    # Recency Boost
    qualified_movies = _apply_recency_boost(qualified_movies)

    # Combine recency with final_score
    qualified_movies["final_score"] = (
        qualified_movies["final_score"] * (1 - recency_weight)
        + qualified_movies["recency_score"] * recency_weight
    )

    # Sort & Return
    return qualified_movies.sort_values("final_score", ascending=False).head(
        min(top_n, len(qualified_movies))
    )[
        [
            "id",
            "title",
            "release_date",
            "final_score",
        ]
    ]


def get_top_similar_movies(movie_index, count_matrix=count_matrix, top_n=62):
    # Compute similarity scores dynamically (only one movie vector at a time)
    movie_vector = count_matrix[movie_index]
    similarity_scores = cosine_similarity(movie_vector, count_matrix).flatten()

    # Get indices of top similar movies (excluding the movie itself)
    similar_indices = similarity_scores.argsort()[::-1][1 : top_n + 1]

    # Return similar movie indices and their similarity scores
    return similar_indices, similarity_scores[similar_indices]


def _apply_recency_boost(df):
    """
    Helper function to parse release_date into a numeric year,
    then create a 0–1 scaled 'recency_score' column in df.
    """
    # Parse year from release_date
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    # Fallback for rows with missing/invalid release_date
    df["year"] = df["year"].fillna(df["year"].min())

    # Scale years to [0,1] range
    min_year = df["year"].min()
    max_year = df["year"].max()
    year_range = max_year - min_year if max_year != min_year else 1

    df["recency_score"] = (df["year"] - min_year) / year_range

    return df


def improved_hybrid_recommendations(
    user_id,
    title,
    best_svd_model=best_svd1,
    new_df=new_df,
    ratings_df=ratings_df,
    links_df=links_df,
    top_n=10,
    popularity_weight=0.15,
    similarity_weight=0.85,
    recency_weight=0.2,
):
    """
    Generate hybrid recommendations combining content-based and collaborative filtering.

    Parameters:
    user_id (int): User ID for whom recommendations are generated.
    title (str): Movie title used for reference.
    top_n (int): Number of top recommendations (default 10).
    popularity_weight (float): Weight of IMDb rating (default 0.15).
    similarity_weight (float): Weight of content similarity (default 0.85).

    Returns:
    DataFrame: Top N recommended movies sorted by hybrid score.
    """

    # Validate if title exists
    index = indices.get(title, None)
    if index is None:
        raise ValueError(f"Movie title '{title}' not found in the dataset.")

    # Get top content-based similar movies
    similar_movie_indices, similarity_scores = get_top_similar_movies(
        index, count_matrix, top_n=62
    )

    recommended_movies = new_df.iloc[similar_movie_indices][
        ["title", "id", "vote_count", "vote_average", "release_date"]
    ].copy()

    recommended_movies["vote_count"] = (
        recommended_movies["vote_count"].fillna(0).astype(int)
    )
    recommended_movies["vote_average"] = (
        recommended_movies["vote_average"].fillna(0).astype(float)
    )

    # Compute IMDb weighted rating
    C = recommended_movies["vote_average"].mean()
    m = recommended_movies["vote_count"].quantile(0.65)
    recommended_movies["weighted_rating"] = recommended_movies.apply(
        lambda x: weighted_rating(x, C, m), axis=1
    )

    # # Filter low vote count movies
    # qualified_movies = recommended_movies[recommended_movies["vote_count"] >= m].copy()

    qualified_movies = recommended_movies.copy()
    qualified_movies.reset_index(drop=True, inplace=True)

    # Merge similarity scores
    similarity_df = pd.DataFrame(
        {
            "id": new_df.iloc[similar_movie_indices]["id"].values,
            "similarity_score": similarity_scores,
        }
    )

    qualified_movies = recommended_movies.merge(similarity_df, on="id", how="left")
    qualified_movies["similarity_score"] = qualified_movies["similarity_score"].fillna(
        qualified_movies["similarity_score"].min()
    )

    # Cold-start handling for new users
    if user_id not in ratings_df["userId"].unique():
        # Incorporate recency
        qualified_movies = _apply_recency_boost(qualified_movies)

        return handle_new_user(
            qualified_movies,
            top_n,
            popularity_weight,
            similarity_weight,
            recency_weight,
        )

    # Match MovieLens IDs before using SVD
    qualified_movies = match_tmdb_to_movielens(qualified_movies, links_df)

    # Predict ratings using SVD
    user_ratings_count = ratings_df[ratings_df["userId"] == user_id].shape[0]
    qualified_movies = predict_ratings(
        user_id, qualified_movies, best_svd_model, ratings_df, C
    )

    # Compute hybrid recommendation score
    qualified_movies = compute_hybrid_score(
        qualified_movies, user_ratings_count, recency_weight, top_n
    )

    return qualified_movies


# TMDb API Key
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def get_movie_poster(movie_id):
    """
    Fetch movie poster URL from TMDb API.

    Parameters:
    movie_id (int): The ID of the movie for which the poster is to be fetched.

    Returns:
    str: The complete URL of the movie poster if available, otherwise None.
    """
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()

    if "poster_path" in data and data["poster_path"]:
        return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    return None
