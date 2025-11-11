from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Response, Session
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

CSV_URL = (
    "https://dl.dropboxusercontent.com/scl/fi/rcg7hpyxd7z9pwtd1vdlq/"
    "ratings_df.csv?rlkey=oopp2pdgvyink2o8p5nn49uf6&st=kis0a2xb"
)
DEFAULT_YEAR = 2000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
PAGE_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT = 15
CF_WEIGHT = 0.7
CB_WEIGHT = 0.3


def load_base_dataset(csv_url: str = CSV_URL) -> pd.DataFrame:
    """Load the ratings DataFrame from the hosted CSV."""
    df = pd.read_csv(csv_url)
    expected_columns = {
        "movie_id",
        "movie_title",
        "genres",
        "original_language",
        "year_released",
        "user_id",
        "rating_val",
    }
    missing_cols = expected_columns - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset at {csv_url} is missing required columns: {sorted(missing_cols)}"
        )
    return df


def convert_stars(stars: str) -> int:
    """Convert a Letterboxd star string (e.g. '★★★½') to a 0-10 scale."""
    result = 0
    for char in stars:
        if char == "★":
            result += 2
        elif char == "½":
            result += 1
    return result


def _fetch(session: Session, url: str) -> Response:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    if response.status_code == 404:
        raise ValueError(f"Resource not found: {url}")
    response.raise_for_status()
    return response


def get_watched_movies(username: str, delay_seconds: float = PAGE_DELAY_SECONDS) -> List[Tuple[str, int]]:
    """Scrape Letterboxd for the given user's watched movies and ratings."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    first_page_url = f"https://letterboxd.com/{username}/films/page/1/"
    first_response = _fetch(session, first_page_url)
    first_soup = BeautifulSoup(first_response.text, "html.parser")

    page_numbers = [
        int(a.text.strip()) for a in first_soup.select("div.paginate-pages li a") if a.text.strip().isdigit()
    ]
    last_page = max(page_numbers) if page_numbers else 1

    movies: List[Tuple[str, int]] = []

    for page in range(1, last_page + 1):
        url = f"https://letterboxd.com/{username}/films/page/{page}/"
        page_response = _fetch(session, url)
        soup = BeautifulSoup(page_response.text, "html.parser")

        for li_tag in soup.select("li.griditem"):
            div_tag = li_tag.find("div", class_="react-component")
            rating_tag = li_tag.select_one("p.poster-viewingdata span.rating")
            rating_text = rating_tag.text.strip() if rating_tag else None

            if not rating_text:
                continue

            rating_val = convert_stars(rating_text)
            if rating_val == 0:
                continue

            if div_tag:
                slug = div_tag.get("data-item-slug")
                if slug:
                    movies.append((slug, rating_val))

        if page < last_page and delay_seconds > 0:
            time.sleep(delay_seconds)

    if not movies:
        raise ValueError(
            f"No rated movies found for user '{username}'. The profile may be private or unrated."
        )

    return movies


def integrate_user_ratings(
    df: pd.DataFrame, username: str, watched_movies: Iterable[Tuple[str, int]]
) -> pd.DataFrame:
    """Append the scraped user ratings to the dataset and refresh indices."""
    user_df = pd.DataFrame(watched_movies, columns=["movie_id", "rating_val"])
    user_df["user_id"] = username

    df_without_user = df[df["user_id"] != username].copy()

    metadata_cols = [
        "movie_id",
        "movie_title",
        "genres",
        "original_language",
        "year_released",
    ]
    unique_movies = df_without_user[metadata_cols].drop_duplicates("movie_id")

    merged = pd.merge(user_df, unique_movies, on="movie_id", how="left")

    merged["movie_title"] = merged["movie_title"].fillna(
        merged["movie_id"].str.replace("-", " ").str.title()
    )
    merged["genres"] = merged["genres"].fillna("Others")
    merged["original_language"] = merged["original_language"].fillna("en")
    merged["year_released"] = merged["year_released"].fillna(DEFAULT_YEAR).astype(int)

    merged["genre"] = merged["genres"].apply(
        lambda g: g.split(",")[0].strip() if isinstance(g, str) and g.strip() else "Others"
    )

    merged = merged[
        [
            "movie_id",
            "movie_title",
            "genres",
            "original_language",
            "year_released",
            "user_id",
            "rating_val",
            "genre",
        ]
    ]

    combined = pd.concat([df_without_user, merged], ignore_index=True)

    combined["user_idx"] = combined["user_id"].astype("category").cat.codes
    combined["movie_idx"] = combined["movie_id"].astype("category").cat.codes

    return combined


def build_model_artifacts(df: pd.DataFrame) -> Dict[str, object]:
    """Create matrices and lookup tables required for recommendations."""
    user_item_matrix = df.pivot_table(
        index="user_idx", columns="movie_idx", values="rating_val", fill_value=0
    )

    unique_movies = df.drop_duplicates("movie_idx")[["movie_idx", "movie_id", "movie_title", "genres"]].copy()

    genres_for_vector = (
        unique_movies["genres"].fillna("Others").astype(str).str.replace(r"\s*,\s*", "|", regex=True)
    )

    vectorizer = CountVectorizer(
        tokenizer=lambda text: [token.strip() for token in text.split("|") if token.strip()],
        token_pattern=None,
    )
    genre_matrix = vectorizer.fit_transform(genres_for_vector)

    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    if genre_matrix.shape[0] > 0:
        knn.fit(genre_matrix)

    movie_idx_to_row = dict(zip(unique_movies["movie_idx"], range(len(unique_movies))))
    row_to_movie_idx = {row: idx for idx, row in movie_idx_to_row.items()}

    user_mapping = (
        df[["user_id", "user_idx"]].drop_duplicates().set_index("user_id")["user_idx"].to_dict()
    )

    movie_info = unique_movies.set_index("movie_idx")[
        ["movie_title", "genres"]
    ]

    return {
        "df": df,
        "user_item_matrix": user_item_matrix,
        "knn": knn,
        "genre_matrix": genre_matrix,
        "movie_idx_to_row": movie_idx_to_row,
        "row_to_movie_idx": row_to_movie_idx,
        "user_mapping": user_mapping,
        "movie_info": movie_info,
    }


def get_similar_users(
    user_idx: int, user_item_matrix: pd.DataFrame, top_k: int = 10
) -> pd.DataFrame:
    """Find the most similar users using cosine similarity."""
    user_vector = user_item_matrix.loc[[user_idx]]
    similarities = cosine_similarity(user_vector, user_item_matrix)[0]

    sim_df = pd.DataFrame({"user_idx": user_item_matrix.index, "similarity": similarities})
    sim_df = sim_df[sim_df["user_idx"] != user_idx]
    return sim_df.sort_values("similarity", ascending=False).head(top_k)


def get_similar_movies_knn(
    movie_idx: int,
    artifacts: Dict[str, object],
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    movie_idx_to_row = artifacts["movie_idx_to_row"]
    genre_matrix = artifacts["genre_matrix"]
    knn: NearestNeighbors = artifacts["knn"]

    if movie_idx not in movie_idx_to_row:
        return []
    if genre_matrix.shape[0] <= 1:
        return []

    row_idx = movie_idx_to_row[movie_idx]
    n_neighbors = min(top_k + 1, genre_matrix.shape[0])
    distances, indices = knn.kneighbors(genre_matrix[row_idx], n_neighbors=n_neighbors)

    similar: List[Tuple[int, float]] = []
    for dist, idx in zip(distances[0][1:], indices[0][1:]):
        neighbor_movie_idx = artifacts["row_to_movie_idx"].get(idx)
        if neighbor_movie_idx is not None:
            similar.append((neighbor_movie_idx, 1 - dist))
    return similar


def get_cb_recommendations(
    user_idx: int,
    artifacts: Dict[str, object],
    top_n: int = 10,
) -> pd.DataFrame:
    user_item_matrix: pd.DataFrame = artifacts["user_item_matrix"]
    user_ratings = user_item_matrix.loc[user_idx]
    rated_indices = user_ratings[user_ratings > 0].index

    scores: defaultdict[int, float] = defaultdict(float)

    for movie in rated_indices:
        rating = user_ratings[movie]
        neighbors = get_similar_movies_knn(movie, artifacts, top_k=50)
        for neighbor_movie, similarity in neighbors:
            if neighbor_movie in rated_indices:
                continue
            scores[neighbor_movie] += similarity * (rating / 10.0)

    if not scores:
        return pd.DataFrame(columns=["movie_idx", "cb_score"])

    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(top_movies, columns=["movie_idx", "cb_score"])


def get_cf_recommendations(
    user_idx: int,
    similar_users: pd.DataFrame,
    artifacts: Dict[str, object],
    top_n: int = 10,
) -> pd.DataFrame:
    if similar_users.empty:
        return pd.DataFrame(columns=["movie_idx", "cf_score"])

    user_item_matrix: pd.DataFrame = artifacts["user_item_matrix"]
    user_seen = set(user_item_matrix.columns[user_item_matrix.loc[user_idx] > 0])

    scores: defaultdict[int, float] = defaultdict(float)
    for _, row in similar_users.iterrows():
        neighbor_idx = row["user_idx"]
        similarity = row["similarity"]
        if similarity <= 0:
            continue

        neighbor_ratings = user_item_matrix.loc[neighbor_idx]
        for movie, rating in neighbor_ratings.items():
            if rating <= 0 or movie in user_seen:
                continue
            scores[movie] += similarity * rating

    if not scores:
        return pd.DataFrame(columns=["movie_idx", "cf_score"])

    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(top_movies, columns=["movie_idx", "cf_score"])


def hybrid_recommend(
    username: str,
    artifacts: Dict[str, object],
    top_n: int = 10,
    cf_weight: float = CF_WEIGHT,
    cb_weight: float = CB_WEIGHT,
) -> pd.DataFrame:
    user_mapping: Dict[str, int] = artifacts["user_mapping"]
    if username not in user_mapping:
        raise ValueError(f"User '{username}' not present in dataset after integration.")

    user_idx = user_mapping[username]
    user_item_matrix: pd.DataFrame = artifacts["user_item_matrix"]

    similar_users = get_similar_users(user_idx, user_item_matrix, top_k=10)
    cf_df = get_cf_recommendations(user_idx, similar_users, artifacts, top_n=top_n * 2)
    cb_df = get_cb_recommendations(user_idx, artifacts, top_n=top_n * 2)

    hybrid_df = pd.merge(cf_df, cb_df, on="movie_idx", how="outer").fillna(0.0)
    if hybrid_df.empty:
        return hybrid_df

    hybrid_df["hybrid_score"] = (
        cf_weight * hybrid_df.get("cf_score", 0) + cb_weight * hybrid_df.get("cb_score", 0)
    )
    hybrid_df = hybrid_df.sort_values("hybrid_score", ascending=False).head(top_n)

    movie_info: pd.DataFrame = artifacts["movie_info"]
    result = hybrid_df.merge(movie_info, left_on="movie_idx", right_index=True)
    return result[["movie_title", "genres", "hybrid_score", "cf_score", "cb_score"]]


def recommend_for_user(
    username: str,
    csv_url: str = CSV_URL,
    top_n: int = 10,
    delay_seconds: float = PAGE_DELAY_SECONDS,
) -> pd.DataFrame:
    base_df = load_base_dataset(csv_url)
    watched_movies = get_watched_movies(username, delay_seconds=delay_seconds)
    combined_df = integrate_user_ratings(base_df, username, watched_movies)
    artifacts = build_model_artifacts(combined_df)
    return hybrid_recommend(username, artifacts, top_n=top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape a Letterboxd user and generate hybrid movie recommendations "
            "using the pre-built ratings dataset."
        )
    )
    parser.add_argument("username", help="Letterboxd username to recommend for")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10)",
    )
    parser.add_argument(
        "--csv-url",
        default=CSV_URL,
        help="Override the ratings CSV URL (default is the hosted Dropbox link)",
    )
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="Disable polite delay between scraping requests",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=PAGE_DELAY_SECONDS,
        help="Seconds to wait between pagination requests (default: 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delay = 0.0 if args.no_delay else max(args.delay, 0.0)

    recommendations = recommend_for_user(
        username=args.username,
        csv_url=args.csv_url,
        top_n=args.top,
        delay_seconds=delay,
    )

    if recommendations.empty:
        print("No recommendations could be generated.")
        return

    for _, row in recommendations.iterrows():
        title = row["movie_title"]
        genres = re.sub(r"\s{2,}", " ", str(row["genres"]))
        score = row["hybrid_score"]
        print(f"{title} — {genres} (score: {score:.2f})")


if __name__ == "__main__":
    main()
