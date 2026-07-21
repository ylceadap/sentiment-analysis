FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY --chown=app:app artifacts/model.joblib artifacts/model_metadata.json artifacts/model_release.json ./artifacts/
COPY --chown=app:app artifacts/final_models/comparison.json ./artifacts/final_models/comparison.json

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "dutch_sentiment.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
