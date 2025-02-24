import numpy as np
import pandas as pd
from surprise import PredictionImpossible
import pickle


def load_data():
    """
    Load datasets for MovieLens, TMDb links, and movie metadata.

    Returns:
    tuple: DataFrames containing ratings, links, and merged movie details.
    """
    ratings_df = pd.read_csv(r"data/ratings_small.csv")  # MovieLens ratings
    links_df = pd.read_csv(r"data/links_small.csv")  # TMDb and MovieLens ID mapping
    new_df = pd.read_csv(r"data/merged_df.csv")  # Movie details

    return ratings_df, links_df, new_df


def load_similarity_matrix():
    """
    Load the precomputed content-based cosine similarity matrix.

    Returns:
    numpy.ndarray: Cosine similarity matrix.
    """
    return np.load(r"data/cosine_similarity2.npy")


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
cosine_similarity2 = load_similarity_matrix()
best_svd1 = load_model()

# Create Movie Index Mapping (Title to Index)
indices = pd.Series(new_df.index, index=new_df["title_x"]).drop_duplicates()


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


def handle_new_user(qualified_movies, top_n, popularity_weight, similarity_weight):
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

    # Compute Final Score for Sorting
    qualified_movies["final_score"] = (
        popularity_weight * qualified_movies["weighted_rating"]
    ) + (similarity_weight * qualified_movies["similarity_score"])
    return qualified_movies.sort_values("final_score", ascending=False).head(top_n)[
        ["id", "title_x", "weighted_rating", "final_score"]
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


def compute_hybrid_score(qualified_movies, user_ratings_count):
    """
    Compute final hybrid recommendation score with dynamic weighting.

    Parameters:
    qualified_movies (DataFrame): Movies selected for recommendation.
    user_ratings_count (int): Number of ratings given by the user.

    Returns:
    DataFrame: Movies DataFrame with hybrid scores.
    """
    if user_ratings_count < 10:
        svd_weight = 0.5  # Less confidence in SVD for new users
    elif user_ratings_count < 50:
        svd_weight = 0.6
    else:
        svd_weight = 0.7  # Higher confidence for active users

    imdb_weight = 1 - svd_weight
    qualified_movies["final_score"] = (
        svd_weight * qualified_movies["predicted_rating"]
    ) + (imdb_weight * qualified_movies["weighted_rating"])
    return qualified_movies


def improved_hybrid_recommendations(
    user_id, title, top_n=10, popularity_weight=0.15, similarity_weight=0.85
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

    # Get top similar movies using content-based filtering
    similarity_scores = np.array(cosine_similarity2[index])
    similar_movie_indices = similarity_scores.argsort()[::-1][
        1:52
    ]  # Get top 50 similar movies (excluding itself)

    recommended_movies = new_df.iloc[similar_movie_indices][
        ["title_x", "id", "vote_count", "vote_average", "release_date", "genres"]
    ]
    recommended_movies = recommended_movies.copy()
    recommended_movies["vote_count"] = (
        recommended_movies["vote_count"].fillna(0).astype(int)
    )
    recommended_movies["vote_average"] = (
        recommended_movies["vote_average"].fillna(0).astype(float)
    )

    # Compute IMDb weighted rating
    C = recommended_movies["vote_average"].mean()
    m = recommended_movies["vote_count"].quantile(0.60)
    recommended_movies["weighted_rating"] = recommended_movies.apply(
        lambda x: weighted_rating(x, C, m), axis=1
    )

    # Filter low vote count movies
    qualified_movies = recommended_movies[recommended_movies["vote_count"] >= m].copy()

    # Reset index to align indices properly
    qualified_movies = qualified_movies.reset_index(drop=True)

    # Keep only the indices that are still present in qualified_movies
    filtered_similarity_scores = pd.DataFrame(
        {
            "id": new_df.iloc[similar_movie_indices]["id"].values,
            "similarity_score": similarity_scores[similar_movie_indices],
        }
    )

    # Merge to assign similarity scores properly
    qualified_movies = qualified_movies.merge(
        filtered_similarity_scores, on="id", how="left"
    )

    # Fill missing similarity scores (if any) with the minimum similarity
    qualified_movies["similarity_score"] = qualified_movies["similarity_score"].fillna(
        qualified_movies["similarity_score"].min()
    )

    # # Ensure the length matches
    # qualified_movies = qualified_movies.loc[filtered_indices]
    # qualified_movies['similarity_score'] = filtered_similarity_scores

    # Match MovieLens IDs before using SVD
    qualified_movies = match_tmdb_to_movielens(qualified_movies, links_df)

    # Cold-start handling for new users
    if user_id not in ratings_df["userId"].unique():
        return handle_new_user(
            qualified_movies, top_n, popularity_weight, similarity_weight
        )

    # Predict ratings using SVD
    user_ratings_count = ratings_df[ratings_df["userId"] == user_id].shape[0]
    qualified_movies = predict_ratings(
        user_id, qualified_movies, best_svd1, ratings_df, C
    )

    # Compute hybrid recommendation score
    qualified_movies = compute_hybrid_score(qualified_movies, user_ratings_count)

    return qualified_movies.sort_values("final_score", ascending=False).head(
        min(top_n, len(qualified_movies))
    )[["id", "title_x", "weighted_rating", "predicted_rating", "final_score"]]
