from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from backend import core_recommender as core


class RecommendationError(Exception):
    """Custom exception for recommendation failures."""


@lru_cache(maxsize=1)
def get_artifacts() -> Dict[str, object]:
    print("🔹 Loading precomputed artifacts from:", core.ARTIFACTS_URL)
    artifacts = core.load_artifacts(core.ARTIFACTS_URL)
    print("✅ Artifacts loaded successfully.")
    return artifacts


def refresh_dataset() -> None:
    get_artifacts.cache_clear()
    get_artifacts()


def generate_recommendations(username: str, top_n: int = 10) -> List[str]:
    if not username or not username.strip():
        raise RecommendationError("Username must be provided.")

    try:
        watched_movies = core.get_watched_movies(username.strip())
    except Exception as exc:  # noqa: BLE001
        raise RecommendationError(str(exc)) from exc

    artifacts = get_artifacts()
    user_vector, seen_indices = core.build_user_vector_from_watched(watched_movies, artifacts)

    if not seen_indices:
        raise RecommendationError(
            "We couldn't match any of this user's movies to our dataset yet. Try another profile."
        )

    try:
        recs_df = core.hybrid_recommend_for_new_user(user_vector, seen_indices, artifacts, top_n=top_n)
    except Exception as exc:  # noqa: BLE001
        raise RecommendationError("Failed to generate recommendations.") from exc

    if recs_df.empty:
        raise RecommendationError("No recommendations available for this user.")

    return recs_df["movie_title"].tolist()
