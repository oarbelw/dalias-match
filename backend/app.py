from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:  # pragma: no cover - import convenience for local vs package execution
    from .recommendation_service import (
        RecommendationError,
        generate_recommendations,
        refresh_dataset,
    )
except ImportError:  # noqa: F401  # pragma: no cover
    from recommendation_service import (  # type: ignore
        RecommendationError,
        generate_recommendations,
        refresh_dataset,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Dalia's Match API", version="1.0.0")

# Allow requests from any origin during development; tighten later if desired
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_dataset() -> None:
    """Eagerly load the ratings dataset into memory."""
    try:
        refresh_dataset()
        logger.info("Ratings dataset loaded and cached successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to preload dataset: %s", exc)


@app.get("/health", summary="API health check")
def health_check() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/recommend", summary="Get movie recommendations")
def recommend(username: str = Query(..., description="Letterboxd username")) -> Dict[str, Any]:
    try:
        recommendations = generate_recommendations(username)
    except RecommendationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while processing user '%s'", username)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return {"username": username, "recommendations": recommendations}
