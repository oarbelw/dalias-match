from __future__ import annotations

from collections import defaultdict
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


def generate_recommendations(username: str, top_n: int = 20) -> List[str]:
    if not username or not username.strip():
        raise RecommendationError("Username must be provided.")

    usernames = [handle.strip() for handle in username.split(",") if handle.strip()]
    if not usernames:
        raise RecommendationError("Username must be provided.")

    artifacts = get_artifacts()
    combined_scores: defaultdict[int, float] = defaultdict(float)
    title_lookup: Dict[int, str] = {}

    for handle in usernames:
        try:
            recs_df = core.recommend_for_user(handle, artifacts=artifacts, top_n=100)
        except ValueError as exc:
            raise RecommendationError(str(exc)) from exc

        for _, row in recs_df.iterrows():
            movie_idx = int(row["movie_idx"])
            combined_scores[movie_idx] += float(row["hybrid_score"])
            if movie_idx not in title_lookup:
                title_lookup[movie_idx] = row["movie_title"]

    if not combined_scores:
        raise RecommendationError("No recommendations available for the provided usernames.")

    top_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [title_lookup[idx] for idx, _ in top_movies]
