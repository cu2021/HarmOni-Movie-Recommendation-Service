from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from my_modules.myModule import improved_hybrid_recommendations, get_movie_poster

# Load environment variables from .env file
load_dotenv()

# Initialize Flask App
app = Flask(__name__)


@app.route("/")
def home():
    """
    Render the homepage.

    Returns:
    HTML page: The index.html template.
    """
    return render_template("index.html")


@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Generate movie recommendations for a given user and movie title.

    Query Parameters:
    userId (int): The ID of the user for whom recommendations are to be generated.
    title (str): The title of the movie for recommendation reference.
    topN (int, optional): The number of top recommendations to return (default is 10).

    Returns:
    JSON: A response containing the user ID, requested title, and a list of recommended movies with their details.
    """
    try:
        userId = int(request.args.get("userId"))
        title = request.args.get("title")
        topN = int(request.args.get("topN", 10))  # Default to 10 recommendations

        if not title:
            return jsonify({"error": "Movie title is required"}), 400

        # Get recommendations
        recommendations = improved_hybrid_recommendations(
            user_id=userId, title=title, top_n=topN
        )

        # Convert DataFrame to JSON and fetch movie posters
        result = recommendations.to_dict(orient="records")

        for movie in result:
            movie["poster_url"] = get_movie_poster(movie)

        return jsonify(
            {
                "status": True,
                "message": "",
                "data": {
                    "userId": userId,
                    "title": title,
                    "recommendedMovies": result,
                },
            }
        )

    except ValueError as ve:
        return jsonify({"status": False, "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": False, "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
