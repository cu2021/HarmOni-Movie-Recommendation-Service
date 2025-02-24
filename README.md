# HarmOni Movie Recommendation Service

HarmOni is a sophisticated movie recommendation service designed to provide personalized movie suggestions based on user preferences and movie popularity. Developed using Python and Flask, it employs a hybrid recommendation system that integrates content-based filtering with Singular Value Decomposition (SVD) collaborative filtering, along with popularity filtering from the Internet Movie Database ([IMDB](https://www.imdb.com/)).

## Features

- **Personalized Recommendations**: Generates tailored movie suggestions by combining content similarity and user preferences.
- **Movie Poster Retrieval**: Fetches movie posters using the TMDb API.
- **Lightweight Flask API**: Delivers recommendations through a RESTful API.
- **User-Friendly Interface**: Provides a simple HTML-based frontend for user interaction and recommendation display.

## Project Structure

The repository is organized as follows:

- `app.py`: Main Flask application file.
- `data/`: Contains datasets and precomputed similarity matrices.
- `models/`: Stores the pre-trained SVD model for collaborative filtering.
- `my_modules/`: Houses core functionalities, including recommendation logic.
- `static/`: Contains frontend assets such as JavaScript and CSS.
- `templates/`: Stores HTML templates for the frontend.
- `venv/`: Virtual environment directory (excluded from Git tracking).
- `requirements.txt`: Lists required Python dependencies for the project.

## Installation

To set up the HarmOni Movie Recommendation Service locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cu2021/HarmOni-Movie-Recommendation-Service.git
   cd HarmOni-Movie-Recommendation-Service
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Rename `.env.example` to `.env`.
   - Update the `.env` file with the necessary environment variables (e.g., `TMDB_API_KEY`).

5. **Run the application**:
   ```bash
   python app.py
   ```
   The service will now be running locally. Access it in your web browser at `http://127.0.0.1:5000/`.

## Usage

- **Homepage**: Enter a User ID and a Movie Title to receive recommendations.
- **API Endpoint**: Use `/recommend` with the following query parameters:
  - `userId`: The user ID.
  - `title`: The movie title.
  - `topN`: The number of recommendations (default: 10).

  Example API request:
  ```
  http://127.0.0.1:5000/recommend?userId=1&title=Inception&topN=10
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

