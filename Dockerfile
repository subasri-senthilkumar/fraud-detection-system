# FastAPI backend image (also used for one-off training: see docker-compose.yml "trainer").
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by XGBoost's OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src ./src
COPY api ./api
COPY scripts ./scripts
# Trained artifacts are baked in for single-image deployments (e.g. Render);
# docker-compose bind-mounts ./models over this so retraining needs no rebuild.
COPY models ./models

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
