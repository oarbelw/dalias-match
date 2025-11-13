const API_BASE_URL = window.API_BASE_URL || "https://dalias-match-989834185609.us-central1.run.app";

const usernameInput = document.getElementById("username");
const submitButton = document.getElementById("submit");
const feedbackEl = document.getElementById("feedback");
const resultsList = document.getElementById("results-list");

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Fetching..." : "Get Recommendations";
  feedbackEl.textContent = isLoading ? "Hang tight — pulling your Letterboxd data." : "";
}

function renderRecommendations(recommendations) {
  resultsList.innerHTML = "";
  if (!recommendations || recommendations.length === 0) {
    feedbackEl.textContent = "No recommendations found for this user yet.";
    return;
  }

  recommendations.forEach((title, index) => {
    const item = document.createElement("li");
    const titleSpan = document.createElement("strong");
    titleSpan.textContent = title;

    const rankSpan = document.createElement("span");
    rankSpan.textContent = `#${index + 1}`;

    item.appendChild(titleSpan);
    item.appendChild(rankSpan);
    resultsList.appendChild(item);
  });
}

async function fetchRecommendations() {
  const username = usernameInput.value.trim();
  if (!username) {
    feedbackEl.textContent = "Enter a Letterboxd username to continue.";
    usernameInput.focus();
    return;
  }

  const apiUrl = `${API_BASE_URL.replace(/\/$/, "")}/recommend?username=${encodeURIComponent(username)}`;

  setLoading(true);
  try {
    const response = await fetch(apiUrl, { method: "GET" });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const message = errorData.detail || "Unable to fetch recommendations.";
      throw new Error(message);
    }

    const data = await response.json();
    feedbackEl.textContent = `Top picks for @${data.username}`;
    renderRecommendations(data.recommendations);
  } catch (error) {
    console.error(error);
    feedbackEl.textContent = error.message || "Something went wrong.";
    resultsList.innerHTML = "";
  } finally {
    setLoading(false);
  }
}

submitButton.addEventListener("click", fetchRecommendations);
usernameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    fetchRecommendations();
  }
});
