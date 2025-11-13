from __future__ import annotations

import argparse
import io
import os
import pickle
import re
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Response, Session
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

DEFAULT_DATASET_URL = (
    "https://github.com/oarbelw/dalias-match/releases/download/v1.1/ratings_df.parquet"
)
DATASET_URL = os.getenv("CSV_URL", DEFAULT_DATASET_URL)
DEFAULT_ARTIFACTS_URL = (
    "https://github.com/oarbelw/dalias-match/releases/download/v1.2/artifacts.pkl"
)
ARTIFACTS_URL = os.getenv("ARTIFACTS_URL", DEFAULT_ARTIFACTS_URL)
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


def load_base_dataset(csv_url: str = DATASET_URL) -> pd.DataFrame:
    try:
        response = requests.get(csv_url, timeout=60)
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network failure
        raise ValueError(
            "Failed to download ratings dataset. Verify that CSV_URL/ARTIFACTS_URL env vars are correct and publicly accessible."
        ) from exc

    df = pd.read_parquet(io.BytesIO(response.content))
    base_columns = [
        "movie_id",
        "movie_title",
        "genres",
        "original_language",
        "year_released",
        "user_id",
        "rating_val",
    ]
    missing_cols = set(base_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset at {csv_url} is missing required columns: {sorted(missing_cols)}"
        )

    optional_cols = [col for col in ("genre", "user_idx", "movie_idx") if col in df.columns]
    df = df[base_columns + optional_cols]

    df["rating_val"] = df["rating_val"].astype("float32")
    df["year_released"] = df["year_released"].astype("float32")

    return df


def convert_stars(stars: str) -> int:
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
    combined["rating_val"] = combined["rating_val"].astype("float32")
    return combined


def build_model_artifacts(df: pd.DataFrame) -> Dict[str, object]:
    df = df.copy()
    df["rating_val"] = df["rating_val"].astype("float32")

    user_cat = df["user_id"].astype("category")
    movie_cat = df["movie_id"].astype("category")

    df["user_idx"] = user_cat.cat.codes.astype("int32")
    df["movie_idx"] = movie_cat.cat.codes.astype("int32")

    n_users = user_cat.cat.categories.size
    n_movies = movie_cat.cat.categories.size

    ratings = df["rating_val"].to_numpy(dtype="float32")
    user_indices = df["user_idx"].to_numpy(dtype="int32")
    movie_indices = df["movie_idx"].to_numpy(dtype="int32")

    user_item_matrix = csr_matrix((ratings, (user_indices, movie_indices)), shape=(n_users, n_movies))

    user_ratings_map: Dict[int, List[Tuple[int, float]]] = {}
    for user_idx, group in df.groupby("user_idx"):
        user_ratings_map[int(user_idx)] = list(
            zip(group["movie_idx"].astype(int), group["rating_val"].astype(float))
        )

    user_seen_map = {user_idx: {movie for movie, _ in items} for user_idx, items in user_ratings_map.items()}

    unique_movies = df.drop_duplicates("movie_idx")[["movie_idx", "movie_title", "genres"]].copy()

    vectorizer = CountVectorizer(
        tokenizer=lambda text: [token.strip() for token in text.split("|") if token.strip()],
        token_pattern=None,
    )
    genres_for_vector = unique_movies["genres"].fillna("Others").astype(str).str.replace(
        r"\s*,\s*", "|", regex=True
    )
    genre_matrix = vectorizer.fit_transform(genres_for_vector)

    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    if genre_matrix.shape[0] > 0:
        knn.fit(genre_matrix)

    movie_idx_to_row = dict(zip(unique_movies["movie_idx"], range(len(unique_movies))))
    row_to_movie_idx = {row: idx for idx, row in movie_idx_to_row.items()}

    movie_categories = list(movie_cat.cat.categories)
    movie_id_to_idx = {movie_id: int(idx) for idx, movie_id in enumerate(movie_categories)}
    idx_to_movie_id = {int(idx): movie_id for idx, movie_id in enumerate(movie_categories)}

    user_mapping = {
        user_id: int(idx)
        for idx, user_id in enumerate(user_cat.cat.categories)
    }

    movie_info = unique_movies.set_index("movie_idx")[["movie_title", "genres"]]

    return {
        "user_item_matrix": user_item_matrix,
        "user_ratings_map": user_ratings_map,
        "user_seen_map": user_seen_map,
        "knn": knn,
        "genre_matrix": genre_matrix,
        "movie_idx_to_row": movie_idx_to_row,
        "row_to_movie_idx": row_to_movie_idx,
        "movie_id_to_idx": movie_id_to_idx,
        "idx_to_movie_id": idx_to_movie_id,
        "movie_info": movie_info,
    }


def load_artifacts(url: str = ARTIFACTS_URL) -> Dict[str, object]:
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network failure
        raise ValueError(
            "Failed to download recommender artifacts. Ensure ARTIFACTS_URL points to a valid pickle file."
        ) from exc

    return pickle.load(io.BytesIO(response.content))


def build_user_vector_from_watched(
    watched_movies: Iterable[Tuple[str, int]], artifacts: Dict[str, object]
) -> Tuple[np.ndarray, Set[int]]:
    movie_id_to_idx = artifacts.get("movie_id_to_idx", {})
    n_movies = artifacts["user_item_matrix"].shape[1]

    user_vector = np.zeros((1, n_movies), dtype=np.float32)
    seen_indices: Set[int] = set()

    for movie_id, rating in watched_movies:
        movie_idx = movie_id_to_idx.get(movie_id)
        if movie_idx is None:
            continue
        user_vector[0, movie_idx] = float(rating)
        seen_indices.add(movie_idx)

    return user_vector, seen_indices


def get_similar_users_for_vector(
    user_vector: np.ndarray,
    artifacts: Dict[str, object],
    top_k: int = 20,
) -> pd.DataFrame:
    user_item_matrix: csr_matrix = artifacts["user_item_matrix"]
    if user_item_matrix.shape[0] == 0:
        return pd.DataFrame(columns=["user_idx", "similarity"])

    similarities = cosine_similarity(user_vector, user_item_matrix).ravel()
    if np.isnan(similarities).all():
        return pd.DataFrame(columns=["user_idx", "similarity"])

    similarities = np.nan_to_num(similarities)
    top_k = min(top_k, similarities.size)
    if top_k == 0:
        return pd.DataFrame(columns=["user_idx", "similarity"])

    indices = np.argpartition(similarities, -top_k)[-top_k:]
    indices = indices[np.argsort(similarities[indices])[::-1]]

    data = pd.DataFrame({"user_idx": indices, "similarity": similarities[indices]})
    return data[data["similarity"] > 0]


def get_cf_recommendations_for_vector(
    similar_users: pd.DataFrame,
    seen_indices: Set[int],
    artifacts: Dict[str, object],
    top_n: int = 10,
) -> pd.DataFrame:
    if similar_users.empty:
        return pd.DataFrame(columns=["movie_idx", "cf_score"])

    user_ratings_map: Dict[int, List[Tuple[int, float]]] = artifacts["user_ratings_map"]

    scores: defaultdict[int, float] = defaultdict(float)
    for _, row in similar_users.iterrows():
        neighbor_idx = int(row["user_idx"])
        similarity = float(row["similarity"])
        if similarity <= 0:
            continue
        for movie_idx, rating in user_ratings_map.get(neighbor_idx, []):
            if movie_idx in seen_indices or rating <= 0:
                continue
            scores[movie_idx] += similarity * rating

    if not scores:
        return pd.DataFrame(columns=["movie_idx", "cf_score"])

    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(top_movies, columns=["movie_idx", "cf_score"])


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


def get_cb_recommendations_for_vector(
    user_vector: np.ndarray,
    seen_indices: Set[int],
    artifacts: Dict[str, object],
    top_n: int = 10,
) -> pd.DataFrame:
    rated_indices = np.flatnonzero(user_vector[0])
    if rated_indices.size == 0:
        return pd.DataFrame(columns=["movie_idx", "cb_score"])

    scores: defaultdict[int, float] = defaultdict(float)

    for movie_idx in rated_indices:
        rating = float(user_vector[0, movie_idx])
        neighbors = get_similar_movies_knn(movie_idx, artifacts, top_k=50)
        for neighbor_movie, similarity in neighbors:
            if neighbor_movie in seen_indices:
                continue
            scores[neighbor_movie] += similarity * (rating / 10.0)

    if not scores:
        return pd.DataFrame(columns=["movie_idx", "cb_score"])

    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(top_movies, columns=["movie_idx", "cb_score"])


def hybrid_recommend_for_new_user(
    user_vector: np.ndarray,
    seen_indices: Set[int],
    artifacts: Dict[str, object],
    top_n: int = 10,
    cf_weight: float = CF_WEIGHT,
    cb_weight: float = CB_WEIGHT,
) -> pd.DataFrame:
    similar_users = get_similar_users_for_vector(user_vector, artifacts, top_k=20)
    cf_df = get_cf_recommendations_for_vector(similar_users, seen_indices, artifacts, top_n=top_n * 2)
    cb_df = get_cb_recommendations_for_vector(user_vector, seen_indices, artifacts, top_n=top_n * 2)

    hybrid_df = pd.merge(cf_df, cb_df, on="movie_idx", how="outer").fillna(0.0)
    if hybrid_df.empty:
        return hybrid_df

    hybrid_df["hybrid_score"] = (
        cf_weight * hybrid_df.get("cf_score", 0)
        + cb_weight * hybrid_df.get("cb_score", 0)
    )
    hybrid_df = hybrid_df.sort_values("hybrid_score", ascending=False).head(top_n)

    movie_info: pd.DataFrame = artifacts["movie_info"]
    result = hybrid_df.merge(movie_info, left_on="movie_idx", right_index=True)
    return result[["movie_title", "genres", "hybrid_score", "cf_score", "cb_score"]]


def recommend_for_user(
    username: str,
    artifacts: Dict[str, object] | None = None,
    top_n: int = 10,
    delay_seconds: float = PAGE_DELAY_SECONDS,
) -> pd.DataFrame:
    if not username or not username.strip():
        raise ValueError("Username must be provided.")

    watched_movies = get_watched_movies(username.strip(), delay_seconds=delay_seconds)

    if artifacts is None:
        artifacts = load_artifacts()

    user_vector, seen_indices = build_user_vector_from_watched(watched_movies, artifacts)
    if not seen_indices:
        raise ValueError(
            "We couldn't match any of this user's movies to our dataset. Try another profile."
        )

    recommendations = hybrid_recommend_for_new_user(user_vector, seen_indices, artifacts, top_n=top_n)
    if recommendations.empty:
        raise ValueError("No recommendations available for this user.")

    return recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hybrid movie recommendations for a Letterboxd user using precomputed artifacts."
    )
    parser.add_argument("username", help="Letterboxd username to recommend for")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10)",
    )
    parser.add_argument(
        "--artifacts-url",
        default=ARTIFACTS_URL,
        help="Override the artifacts pickle URL (default: value from ARTIFACTS_URL env or release)",
    )
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="Disable polite delay between Letterboxd scraping requests",
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

    artifacts = load_artifacts(args.artifacts_url)

    recommendations = recommend_for_user(
        username=args.username,
        artifacts=artifacts,
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
