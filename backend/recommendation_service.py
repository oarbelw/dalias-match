from __future__ import annotations

from functools import lru_cache
from typing import List

import pandas as pd

from .core_recommender import (
    CSV_URL,
    build_model_artifacts,
    get_watched_movies,
    hybrid_recommend,
    integrate_user_ratings,
    load_base_dataset,
)


class RecommendationError(Exception):
    """Custom exception for recommendation failures."""


@lru_cache(maxsize=1)
def _load_cached_dataset(csv_url: str = CSV_URL) -> pd.DataFrame:
    """Load and cache the base ratings dataset from Dropbox."""
    return load_base_dataset(csv_url)


def refresh_dataset(csv_url: str = CSV_URL) -> None:
    """Clear the dataset cache; useful for manual refreshes."""
    _load_cached_dataset.cache_clear()
    _load_cached_dataset(csv_url)


def generate_recommendations(username: str, top_n: int = 10) -> List[str]:
    """Generate movie recommendations for a Letterboxd username."""
    if not username or not username.strip():
        raise RecommendationError("Username must be provided.")

    try:
        watched_movies = get_watched_movies(username.strip())
    except Exception as exc:  # noqa: BLE001
        raise RecommendationError(str(exc)) from exc

    base_df = _load_cached_dataset().copy()

    try:
        combined_df = integrate_user_ratings(base_df, username, watched_movies)
        artifacts = build_model_artifacts(combined_df)
        recs_df = hybrid_recommend(username, artifacts, top_n=top_n)
    except Exception as exc:  # noqa: BLE001
        raise RecommendationError("Failed to generate recommendations.") from exc

    if recs_df.empty:
        raise RecommendationError("No recommendations available for this user.")

    return recs_df["movie_title"].tolist()
