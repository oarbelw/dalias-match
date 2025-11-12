from __future__ import annotations

from functools import lru_cache
from typing import List

import pandas as pd

import core_recommender as core


class RecommendationError(Exception):
    """Custom exception for recommendation failures."""


@lru_cache(maxsize=1)
def load_dataset(csv_url: str = core.CSV_URL) -> pd.DataFrame:
    """Load and cache the base ratings dataset from the configured source."""
    return core.load_base_dataset(csv_url)


def refresh_dataset(csv_url: str = core.CSV_URL) -> None:
    load_dataset.cache_clear()
    load_dataset(csv_url)


def generate_recommendations(username: str, top_n: int = 10) -> List[str]:
    if not username or not username.strip():
        raise RecommendationError("Username must be provided.")

    try:
        watched_movies = core.get_watched_movies(username.strip())
    except Exception as exc:  # noqa: BLE001
        raise RecommendationError(str(exc)) from exc

    base_df = load_dataset().copy()

    try:
        combined_df = core.integrate_user_ratings(base_df, username, watched_movies)
        artifacts = core.build_model_artifacts(combined_df)
        recs_df = core.hybrid_recommend(username, artifacts, top_n=top_n)
    except Exception as exc:  # noqa: BLE001
        raise RecommendationError("Failed to generate recommendations.") from exc

    if recs_df.empty:
        raise RecommendationError("No recommendations available for this user.")

    return recs_df["movie_title"].tolist()
