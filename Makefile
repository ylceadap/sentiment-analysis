.PHONY: install install-core install-locked install-embeddings audit train embedding-experiment jina-ordinal-logistic evaluate benchmark predict test coverage lint format serve mlflow mlflow-organize docker-build docker-run

PYTHON := .venv/bin/python

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e '.[train,dev]'

install-core:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e .

install-locked:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements/verified-py311.lock
	$(PYTHON) -m pip install --no-deps -e .

install-embeddings:
	$(PYTHON) -m pip install -e '.[train,dev,embeddings]'

audit:
	.venv/bin/sentiment-audit --data Python_Engineer_Challenge_2.csv

train:
	.venv/bin/sentiment-train --config configs/training.yaml

embedding-experiment:
	$(PYTHON) -m dutch_sentiment.experiments.embedding --config configs/models/jina_logreg.yaml

jina-ordinal-logistic:
	$(PYTHON) -m dutch_sentiment.experiments.jina_ordinal --config configs/models/jina_ordinal.yaml

evaluate:
	.venv/bin/sentiment-evaluate

benchmark:
	.venv/bin/sentiment-benchmark --model artifacts/model.joblib

predict:
	@test -n "$(REVIEW)" || (echo "Usage: make predict REVIEW='Deze film was goed.'" && exit 2)
	.venv/bin/sentiment-predict --review "$(REVIEW)"

test:
	.venv/bin/pytest

coverage:
	.venv/bin/pytest --cov=dutch_sentiment --cov-report=term-missing

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:
	.venv/bin/ruff format .

serve:
	.venv/bin/uvicorn dutch_sentiment.api:create_app --factory --host 0.0.0.0 --port 8000

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

mlflow-organize:
	$(PYTHON) scripts/organize_mlflow_registry.py

docker-build:
	docker build -t dutch-sentiment:latest .

docker-run:
	docker run --rm -p 8000:8000 dutch-sentiment:latest
