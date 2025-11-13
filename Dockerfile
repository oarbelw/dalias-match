FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/ backend/
COPY recommender.py recommender.py
COPY frontend/ frontend/

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

RUN python - <<'PY'
from pathlib import Path
import urllib.request

url = "https://github.com/oarbelw/dalias-match/releases/download/v1.1/ratings_df.parquet"
path = Path("backend/data")
path.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, path / "ratings_df.parquet")
PY

ENV PORT=8080

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
