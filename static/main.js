document.addEventListener("DOMContentLoaded", function () {
  const recommendationType = document.getElementById("recommendationType");
  const movieBasedFields = document.getElementById("movieBasedFields");
  const genreBasedFields = document.getElementById("genreBasedFields");

  // Toggle input fields based on recommendation type
  recommendationType.addEventListener("change", function () {
    if (this.value === "movie") {
      movieBasedFields.style.display = "block";
      genreBasedFields.style.display = "none";
    } else {
      movieBasedFields.style.display = "none";
      genreBasedFields.style.display = "block";
    }
  });

  document
    .getElementById("recommendationForm")
    .addEventListener("submit", function (event) {
      event.preventDefault();

      let recommendationType = document.getElementById("recommendationType").value;
      let topN = document.getElementById("topN").value;
      let fetchUrl = "";

      if (recommendationType === "movie") {
        let userId = document.getElementById("userId").value;
        let title = document.getElementById("title").value;
        fetchUrl = `/recommend?userId=${userId}&title=${encodeURIComponent(title)}&topN=${topN}`;
      } else {
        let genre = document.getElementById("genre").value;
        fetchUrl = `/genreBasedRecommendation?genre=${encodeURIComponent(genre)}&topN=${topN}`;
      }

      fetch(fetchUrl)
        .then((response) => response.json())
        .then((data) => {
          if (data.status) {
            let movies = data.data.recommendedMovies;
            let moviesContainer = document.getElementById("moviesGrid");
            moviesContainer.innerHTML = "";

            movies.forEach((movie) => {
              let posterUrl = movie.poster_url
                ? movie.poster_url
                : "https://via.placeholder.com/200x300?text=No+Image";
              let movieCard = `
                  <div class="movie-card">
                      <img src="${posterUrl}" alt="${movie.title}" class="movie-poster">
                      <h3>${movie.title}</h3>
                  </div>
              `;
              moviesContainer.innerHTML += movieCard;
            });
          } else {
            alert("Error: " + data.message);
          }
        })
        .catch((error) => alert("Failed to fetch recommendations."));
    });
});
