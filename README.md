# HarmOni Movie Recommendation Service

HarmOni Movie Recommendation Service is a sophisticated movie recommendation system that provides personalized movie suggestions based on user preferences and popularity. Developed using Python and Flask, it employs a hybrid recommendation approach integrating content-based filtering, Singular Value Decomposition (SVD) collaborative filtering, and popularity filtering from the Internet Movie Database ([IMDB](https://www.imdb.com/)).

## Features

- **Personalized Recommendations**: Generates tailored movie suggestions by combining content similarity and user preferences.
- **Movie Poster Retrieval**: Fetches movie posters using the TMDb API.
- **Lightweight Flask API**: Delivers recommendations through a RESTful API.
- **User-Friendly Interface**: Provides a simple HTML-based frontend for user interaction and recommendation display.
- **Dockerized Deployment**: Easily run the application as a containerized service.

## Project Structure

The repository is organized as follows:

- `app.py`: Main Flask application file.
- `data/`: Contains datasets and precomputed similarity matrices.
- `models/`: Stores the pre-trained SVD model for collaborative filtering.
- `my_modules/`: Houses core functionalities, including recommendation logic.
- `static/`: Contains frontend assets such as JavaScript and CSS.
- `templates/`: Stores HTML templates for the front-end.
- `venv/`: Virtual environment directory (excluded from Git tracking).
- `requirements.txt`: Lists required Python dependencies for the project.
- `Dockerfile`: Defines the containerized environment for deployment.
- `docker-compose.yml`: Configuration for running the service with Docker.

## Installation (Dockerized Deployment)

To set up and run the HarmOni Movie Recommendation Service using Docker, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cu2021/HarmOni-Movie-Recommendation-Service.git
   cd HarmOni-Movie-Recommendation-Service
   ```

2. **Build and Run the Docker Container**:
   ```bash
   docker build -t harmoni-movie-recommendation .
   docker run -p 5000:5000 harmoni-movie-recommendation
   ```
   The service will now be running locally. Access it in your web browser at `http://172.17.0.2:5000/`.

Alternatively, you can use Docker Compose for easier management:

   ```bash
   docker-compose up --build
   ```

## Usage

- **Homepage**: Enter a User ID and a Movie Title to receive recommendations.
- **API Endpoint**:
   - Use `/recommend` with the following query parameters:
     - `userId`: The user ID.
     - `title`: The movie title.
     - `topN`: The number of recommendations (default: 10).

     Example API request:
     ```
     http://172.17.0.2:5000/recommend?userId=1&title=Inception&topN=10
     ```
   - Use `/genreBasedRecommendation` with the following query parameters:
      - `genre`: The genre you like.
      - `topN`:  The number of recommendations (default: 100).
     
     Example API request:
     ```
     http://172.17.0.2:5000/genreBasedRecommendation?genre=Animation&topN=21
     ```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeatureName`.
3. Implement your changes.
4. Commit the changes: `git commit -m 'Add feature: YourFeatureName'`.
5. Push to the branch: `git push origin feature/YourFeatureName`.
6. Open a pull request for review.

## Acknowledgments

Special thanks to the open-source community for their invaluable resources.
