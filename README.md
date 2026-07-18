# Dutch Movie Review Sentiment

A production-minded, CPU-friendly solution to the supplied Python Engineer challenge. It filters confidently Dutch reviews, compares reproducible classical NLP experiments, serves the selected model through FastAPI, and preserves data/model evidence with Git, SHA-256 hashes, MLflow, configuration, and metadata.

The emphasis is engineering method and honest evidence rather than maximizing one accuracy number.

## What is included

- Conservative Unicode/HTML/whitespace normalization inside the serialized sklearn pipeline.
- Local Lingua language identification; no hosted API or LLM dependency.
- Normalized deduplication and deterministic shuffled stratified holdout splitting.
- Five-fold comparison of Dummy, word TF-IDF, character TF-IDF, combined features, class weighting, and rating masking.
- Balanced multinomial Logistic Regression with native probabilities and optional linear feature contributions.
- FastAPI `POST /classify` and `GET /health` endpoints.
- Repeated cold/warm component, service, explanation, and HTTP latency benchmarks.
- Pytest, coverage, Ruff, a non-root Dockerfile, MLflow tracking, and machine-readable model metadata.

## Architecture

```text
raw CSV (immutable)
  -> schema/hash audit
  -> local language identification
  -> normalized deduplication
  -> stratified holdout + stratified training CV
  -> [normalizer -> word/character TF-IDF -> Logistic Regression]
  -> joblib model + JSON metadata
  -> FastAPI inference service
```

Language validation is separate from the sklearn pipeline because it controls request acceptance. Normalization, vectorization, and classification stay in one fitted pipeline so training and serving cannot drift.

## Repository map

```text
configs/training.yaml             central experiment configuration
src/dutch_sentiment/              audit, data, language, model, API, benchmark code
tests/                            deterministic unit and API tests
artifacts/model.joblib            ready-to-serve fitted pipeline
artifacts/model_metadata.json     hashes, versions, split, metrics, schema
artifacts/*.json|*.csv            portable audit/experiment/benchmark evidence
reports/data_audit.md             interpreted full-dataset audit
reports/model_report.md           experiments, test metrics, errors, latency
IMPLEMENTATION_PLAN.md            phase checklist and validation commands
DECISIONS.md                      alternatives, decisions, consequences
REQUIREMENTS_TRACEABILITY.md      requirement-to-evidence mapping
Dockerfile                        non-root serving image
```

## Requirements and installation

- Python 3.11 (verified with 3.11.7); Python 3.12 is allowed by project metadata but was not tested here.
- A Unix-like shell and `make` for convenience; every target shows its underlying command.
- Docker is optional.

```bash
make install
```

Equivalent commands:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Dependencies are constrained by compatible version ranges in `pyproject.toml`. The verified environment resolved scikit-learn 1.9.0, Lingua 2.1.1, FastAPI 0.139.2, MLflow 3.14.0, and pytest 8.4.2.

## Exact working commands

```bash
make audit       # regenerate reports/data_audit.md and artifacts/data_audit.json
make train       # run CV experiments, MLflow tracking, select, fit, and evaluate once
make evaluate    # print stored held-out metrics; does not evaluate the test set again
make benchmark   # regenerate repeated latency evidence and refresh model report
make test        # run pytest
make coverage    # run pytest with branch coverage
make lint        # Ruff lint and formatting checks
make serve       # listen on 0.0.0.0:8000
make mlflow      # open local MLflow UI on port 5000
```

The direct audit command is:

```bash
.venv/bin/sentiment-audit \
  --data Python_Engineer_Challenge_2.csv \
  --output reports/data_audit.md \
  --json-output artifacts/data_audit.json
```

## Data audit: important findings

- 4,800 rows; columns `Reviews` and `Label`; no missing or blank required values.
- Raw SHA-256: `2788b987e2c9fa4fd6459a1798a6e7d1dd63ddb5618c10907712d39042cc4be2`.
- Positive 2,250 (46.88%), Average 2,250 (46.88%), Negative 300 (6.25%).
- Rows are completely label-ordered in three contiguous runs, making sequential splitting invalid.
- Two exact/normalized duplicate extras and no conflicting-label normalized groups.
- 2,908 rows contain HTML breaks, 1,261 contain zero-width characters, 122 match conservative mojibake patterns, and 411 match the documented rating regex.
- Lingua candidates: 4,315 Dutch and 485 English. Non-Dutch removals by original label: 151 Positive, 324 Average, and 10 Negative.
- Maximum source review length is 7,654 characters; the API limit is 8,000 and input is never silently truncated.

Detector output is not treated as annotated ground truth. The bounded manual samples and limitations are in `reports/data_audit.md`.

## Leakage-safe evaluation

After language filtering, two same-label normalized duplicate extras are removed. With seed 42, 3,450 rows form the training/CV partition and 863 form the untouched final test partition. The test supports are Positive 420, Average 385, and Negative 58. Normalized review hashes prove the two partitions are disjoint.

All normalization and TF-IDF fitting happens inside CV folds. The test set is used once after selecting the best mean CV macro-F1.

## Experiment comparison

| Experiment | Features | Balanced | Ratings masked | CV macro-F1 mean ± std | CV balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| combined_balanced_ratings | word + char | yes | no | **0.6544 ± 0.0174** | **0.6218** |
| combined_balanced_masked_ratings | word + char | yes | yes | 0.6507 ± 0.0173 | 0.6169 |
| combined_logreg | word + char | no | no | 0.4946 ± 0.0157 | 0.4888 |
| char_logreg | char | no | no | 0.4716 ± 0.0129 | 0.4739 |
| word_logreg | word | no | no | 0.4427 ± 0.0049 | 0.4572 |
| dummy_prior | ignored baseline features | no | no | 0.2181 ± 0.0002 | 0.3333 |

Full metrics, standard deviations, MLflow run IDs, and artifact sizes are in `artifacts/experiment_comparison.csv`.

### Rating-leakage result

Masking ratings reduced mean macro-F1 by 0.0038, much less than either experiment's fold-to-fold standard deviation. Retaining ratings therefore won the predefined selection metric, but the model does not appear heavily dependent on them. Ratings are legitimate review content and a plausible direct label cue; both interpretations remain documented.

## Selected model and held-out metrics

The selected model combines word unigrams/bigrams and character 3–5-grams with class-balanced Logistic Regression. It was favored for macro-F1, minority-class behavior, native probabilities, size, CPU latency, simple serialization, and inspectable coefficients.

| Metric | Held-out value |
| --- | ---: |
| Accuracy | 0.6477 |
| Balanced accuracy | 0.6055 |
| Macro precision | 0.6709 |
| Macro recall | 0.6055 |
| Macro-F1 | **0.6311** |
| Weighted F1 | 0.6474 |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Positive | 0.6731 | 0.6619 | 0.6675 | 420 |
| Average | 0.6146 | 0.6545 | 0.6340 | 385 |
| Negative | **0.7250** | **0.5000** | **0.5918** | 58 |

Negative is the smallest class; 29/58 held-out Negative reviews were correctly classified, 20 became Average, and 9 became Positive. See `reports/model_report.md` for the full confusion matrix and bounded error analysis.

## Latency evidence

Measured on macOS x86_64, Python 3.11.7, using three short/medium/long Dutch samples, 10 warm-ups, and 100 iterations (HTTP: 50):

| Path | Mean ms | p50 ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| Normalization + model | 3.417 | 2.909 | 5.168 | 5.447 |
| Language detection + model | 4.381 | 3.814 | 6.837 | 7.536 |
| Service end-to-end | 8.095 | 6.778 | 12.258 | 13.686 |
| Explanation enabled | 133.652 | 131.707 | 142.575 | 147.226 |
| HTTP `/classify` | 9.701 | 8.292 | 13.913 | 14.039 |

- Lingua constructor: 0.308 ms, but its lazy first inference/model load: 1,397.708 ms.
- Serialized model load: 599.289 ms; first model prediction: 11.817 ms.
- Artifact: 2,922,196 bytes; SHA-256 `e1eac0ebd0b20133e5537b53c13673b57fca0a32936063739697dd04c96b22b0`.

The API warms Lingua during lifespan startup, so the first production request does not pay the lazy detector cost. Startup/readiness therefore includes that approximately 1.5-second cold cost.

## REST API

Start the API:

```bash
make serve
```

Health:

```bash
curl -sS http://localhost:8000/health
```

```json
{"status":"ok","model_version":"0.1.0+...","model_ready":true}
```

Classify one Dutch review:

```bash
curl -sS -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"review":"Deze film was verrassend goed, met sterke acteurs en een mooi einde.","explain":false}'
```

The response contains one exact allowed label plus native Logistic Regression probabilities:

```json
{
  "label": "Average",
  "model_version": "0.1.0+...",
  "detected_language": "dutch",
  "probabilities": {"Average": 0.542117, "Negative": 0.130673, "Positive": 0.32721},
  "latency_ms": 8.25,
  "explanation": null
}
```

Set `"explain": true` to receive supporting/opposing word n-grams plus separately labeled technical character n-grams. These are linear contributions, not causal explanations.

### Validation and language policy

- Missing, blank, whitespace-only, extra-field, or longer-than-8,000-character input returns HTTP 422.
- Confident non-Dutch input returns HTTP 422:

```json
{"detail":"The review was confidently detected as non-Dutch; submit a Dutch review."}
```

- Text shorter than 20 characters is marked `ambiguous` and allowed through instead of being falsely rejected. This explicitly trades some non-Dutch false acceptance for safer behavior on valid short Dutch input such as “Goed”.
- Unexpected prediction failures return a generic HTTP 500 without the internal exception.
- Logs include request ID, length, detected language, label, latency, and error category—not full review text.

## Tests and quality

Verified commands:

```bash
make test
# 23 passed, 1 third-party Starlette deprecation warning

make coverage
# 57% total branch coverage
# API 91%, audit 79%, data 78%, language 96%, metrics 100%, model 82%, service 97%, text 100%

make lint
# Ruff lint passed; all files formatted
```

The lower aggregate is caused by un-unit-tested CLI orchestration in training/benchmark/report generation; those paths were executed end to end and produced the tracked artifacts. Critical request, transformation, language, model, audit, and metric logic has direct tests.

## MLflow

Training writes live experiment state to a local SQLite backend, ignored by Git, and exports portable run IDs/metrics to CSV and JSON.

```bash
make mlflow
# open http://127.0.0.1:5000
```

The legacy MLflow directory file store was not used because installed MLflow 3.14 puts it in maintenance mode.

## Docker

```bash
make docker-build
make docker-run
```

Then repeat the health and classify curl commands. The image uses Python 3.11 slim, installs runtime dependencies only, runs as a non-root user, includes a health check, and copies the model/metadata but excludes the raw CSV, PDF, tests, reports, Git data, and MLflow database.

**Verification status:** Docker runtime was not verified because no Docker executable/engine is installed in the execution environment. The Dockerfile and build context were reviewed statically; do not interpret this as a successful image build.

## Traceability and security

- Git commits track implementation phases.
- Raw data, split content, fitted model, and package versions are hashed/recorded.
- `artifacts/model_metadata.json` includes the MLflow run, training timestamp, Git commit, language config, label classes, schema, split evidence, and held-out metrics.
- The original CSV remains unchanged and is rehashed during verification.
- `joblib`/pickle artifacts can execute code while loading. Only load the repository's locally controlled artifact; never accept an untrusted uploaded model.
- No secrets or hosted services are required.

## Limitations and sensible next steps

- Lingua can misclassify mixed, translated, short, or named-entity-heavy text.
- Sparse n-grams do not deeply understand sarcasm, negation scope, or long-range composition.
- Negative has only 290 confident Dutch rows and 58 held-out examples.
- Source labels and their possible derivation from ratings were not independently verified.
- There are no timestamps/source IDs for drift or lineage analysis.
- Startup capacity and readiness time should account for Lingua's approximately 1.5-second cold loading cost.
- A compact Dutch transformer is a future experiment only after collecting a larger adjudicated benchmark and defining latency/cost constraints.
