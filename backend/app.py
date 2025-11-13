from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.recommendation_service import (
    RecommendationError,
    generate_recommendations,
    load_dataset,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Dalia's Match API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _ensure_dataset_loaded() -> None:
    """Load the dataset on-demand and cache the result."""
    try:
        load_dataset()
        logger.info("Ratings dataset is ready.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load dataset: %s", exc)
        raise


@app.get("/", summary="Root endpoint")
def root() -> Dict[str, str]:
    return {"message": "Dalia's Match API. Use /health or /recommend?username=<letterboxd_user>."}


@app.get("/health", summary="API health check")
def health_check() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/recommend", summary="Get movie recommendations")
def recommend(username: str = Query(..., description="Letterboxd username")) -> Dict[str, Any]:
    try:
        _ensure_dataset_loaded()
        recommendations = generate_recommendations(username)
    except RecommendationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while processing user '%s'", username)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return {"username": username, "recommendations": recommendations}
