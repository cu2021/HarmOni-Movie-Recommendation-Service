document
.getElementById("recommendationForm")
.addEventListener("submit", function (event) {
  event.preventDefault();

  let userId = document.getElementById("userId").value;
  let title = document.getElementById("title").value;
  let topN = document.getElementById("topN").value;

  fetch(
    `/recommend?userId=${userId}&title=${encodeURIComponent(
      title
    )}&topN=${topN}`
  )
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