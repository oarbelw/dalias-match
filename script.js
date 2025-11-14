const API_BASE =
  window.API_BASE_URL || "https://dalias-match-989834185609.us-central1.run.app";

const usernameInput = document.getElementById("username");
const resultsContainer = document.getElementById("results");
const goButton = document.getElementById("goBtn");

const loadingMarkup = `
  <div class="card" style="grid-column:1/-1;text-align:center;padding:22px">
    <div class="spinner" style="margin:6px auto 10px;width:22px;height:22px;border:3px solid rgba(255,255,255,.2);border-top-color:currentColor;border-radius:50%;animation:spin .8s linear infinite"></div>
    <div>Fetching recommendations…</div>
  </div>
`;

function renderLoading() {
  if (resultsContainer) {
    resultsContainer.innerHTML = loadingMarkup;
  }
}

function renderError(message) {
  if (!resultsContainer) return;
  resultsContainer.innerHTML = `
    <div class="card" style="grid-column:1/-1;border:1px solid rgba(255,0,0,.2)">
      <h3>Oops!</h3>
      <div class="meta">${message}</div>
    </div>
  `;
}

function renderResults(titles) {
  if (!resultsContainer) return;
  if (!Array.isArray(titles) || titles.length === 0) {
    resultsContainer.innerHTML = `
      <div class="card" style="grid-column:1/-1;text-align:center">
        No recommendations yet. Try a different username.
      </div>`;
    return;
  }

  resultsContainer.innerHTML = titles
    .map(
      (title, idx) => `
        <article class="card">
          <h3>#${idx + 1} · ${title}</h3>
          <div class="meta">Recommended for you</div>
        </article>`
    )
    .join("");
}

async function fetchRecommendations(handleInput) {
  const url = `${API_BASE.replace(/\/$/, "")}/recommend?username=${encodeURIComponent(
    handleInput
  )}`;
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return response.json();
}

async function handleSubmit() {
  if (!usernameInput) return;

  const rawValue = usernameInput.value.trim();
  if (!rawValue) {
    renderError("Enter at least one Letterboxd username.");
    return;
  }

  renderLoading();

  try {
    const data = await fetchRecommendations(rawValue);
    renderResults(data?.recommendations ?? []);
  } catch (error) {
    console.error(error);
    renderError("Failed to fetch recommendations. Please try again.");
  }
}

if (goButton) {
  goButton.addEventListener("click", handleSubmit);
}

if (usernameInput) {
  usernameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSubmit();
    }
  });
}