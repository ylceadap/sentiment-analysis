.PHONY: install install-core install-locked install-embeddings audit train embedding-experiment jina-ordinal-logistic final-compare final-compare-log evaluate benchmark predict test coverage lint format serve mlflow mlflow-organize mlflow-audit model-release-verify model-release-export blind-evaluate docker-build docker-run

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

final-compare:
	$(PYTHON) -m dutch_sentiment.final_comparison --config configs/final_five_comparison.yaml --log-mlflow

final-compare-log:
	$(PYTHON) -m dutch_sentiment.final_comparison --config configs/final_five_comparison.yaml --log-existing

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

mlflow-audit:
	$(PYTHON) scripts/organize_mlflow_registry.py --audit-only

model-release-verify:
	$(PYTHON) scripts/manage_model_release.py verify --require-mlflow

model-release-export:
	$(PYTHON) scripts/manage_model_release.py manifest
	$(PYTHON) scripts/manage_model_release.py export

blind-evaluate:
	$(PYTHON) -m dutch_sentiment.blind_evaluation --config configs/blind_evaluation.yaml --confirm-unseen

docker-build:
	docker build -t dutch-sentiment:latest .

docker-run:
	docker run --rm -p 8000:8000 dutch-sentiment:latest
