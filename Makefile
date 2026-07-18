.PHONY: install install-llm audit train llm-experiment evaluate benchmark predict test coverage lint format serve mlflow docker-build docker-run

PYTHON := .venv/bin/python

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e '.[train,dev]'

install-llm:
	$(PYTHON) -m pip install -e '.[train,dev,llm]'

audit:
	.venv/bin/sentiment-audit --data Python_Engineer_Challenge_2.csv

train:
	.venv/bin/sentiment-train --config configs/training.yaml

llm-experiment:
	.venv/bin/python -m dutch_sentiment.llm_experiment --config configs/llm_experiment.yaml --run

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

docker-build:
	docker build -t dutch-sentiment:latest .

docker-run:
	docker run --rm -p 8000:8000 dutch-sentiment:latest
