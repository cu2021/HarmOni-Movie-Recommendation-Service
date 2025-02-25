import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import patch
from app import app

@pytest.fixture
def client():
    """
    Creates a test client to simulate requests to the Flask app.
    
    This fixture provides a `client` object that can be used in test functions
    to send HTTP requests to the application and receive responses. The `client`
    is a Flask test client used for making requests without running the actual app.

    Returns:
        FlaskClient: The test client for making HTTP requests.
    """
    with app.test_client() as client:
        yield client

def test_home(client):
    """
    Tests the home route ("/") of the Flask application.

    This test sends a GET request to the home route and asserts that:
    1. The response status code is 200 (OK).
    2. The response contains the expected string (which can be adjusted based on actual content).
    
    Args:
        client (FlaskClient): The Flask test client.

    Asserts:
        status_code == 200
        response contains "Hybrid Movie Recommendation System"
    """
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hybrid Movie Recommendation System" in response.data 

def test_recommend(client):
    """
    Tests the /recommend route with valid input parameters.

    This test sends a GET request to the /recommend endpoint with a valid userId,
    movie title, and number of recommendations (topN). It asserts that:
    1. The response status code is 200 (OK).
    2. The response contains a valid status of True.
    3. The response contains the expected number of movie recommendations (5 in this case).
    
    Args:
        client (FlaskClient): The Flask test client.

    Asserts:
        status_code == 200
        response contains status == True
        response contains topN recommended movies (5 in this case)
    """
    response = client.get("/recommend?userId=1&title=Inception&topN=5")
    assert response.status_code == 200
    data = response.json
    assert data["status"] is True
    assert len(data["data"]["recommendedMovies"]) == 5 
    
def test_recommend_missing_title(client):
    """
    Tests the /recommend route when the movie title is missing from the request.

    This test sends a GET request to the /recommend endpoint without providing a
    movie title and asserts that:
    1. The response status code is 400 (Bad Request).
    2. The response contains an error message indicating that the movie title is required.
    
    Args:
        client (FlaskClient): The Flask test client.

    Asserts:
        status_code == 400
        response contains "Movie title is required"
    """
    response = client.get("/recommend?userId=1&topN=5")
    assert response.status_code == 400
    assert b"Movie title is required" in response.data

def test_recommend_invalid_userid(client):
    """
    Tests the /recommend route when the userId is invalid.

    This test sends a GET request to the /recommend endpoint with an invalid userId
    (non-integer) and asserts that:
    1. The response status code is 400 (Bad Request).
    2. The response contains an error message indicating an invalid userId.
    
    Args:
        client (FlaskClient): The Flask test client.

    Asserts:
        status_code == 400
        response contains "invalid literal for int()"
    """
    response = client.get("/recommend?userId=abc&title=Inception")
    assert response.status_code == 400
    assert b"invalid literal for int()" in response.data 


def test_get_movie_poster(client):
    """
    Tests the get_movie_poster function by mocking the TMDb API response.

    This test sends a GET request to the /recommend endpoint and mocks the response
    from the TMDb API for fetching a movie poster. It asserts that:
    1. The response status code is 200 (OK).
    2. The movie poster URL in the response matches the expected URL based on the mocked TMDb response.
    
    Args:
        client (FlaskClient): The Flask test client.

    Asserts:
        status_code == 200
        response contains the mocked poster URL
    """
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"poster_path": "/path/to/poster.jpg"}
        response = client.get("/recommend?userId=1&title=Inception&topN=5")
        assert response.status_code == 200
        data = response.json
        assert data["data"]["recommendedMovies"][0]["poster_url"] == "https://image.tmdb.org/t/p/w500/path/to/poster.jpg"
