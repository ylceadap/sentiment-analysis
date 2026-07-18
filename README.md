# Dutch Movie Review Sentiment

A production-minded, CPU-friendly solution to the supplied Python Engineer challenge. It trains one shared Dutch-primary model on every supplied Dutch and English review, compares reproducible classical NLP experiments, serves the selected model through FastAPI, and preserves data/model evidence with Git, SHA-256 hashes, MLflow, configuration, and metadata.

The emphasis is engineering method and honest evidence rather than maximizing one accuracy number.

## What is included

- Conservative Unicode/HTML/whitespace normalization inside the serialized sklearn pipeline.
- Local Lingua language identification; the submitted model has no hosted API or LLM dependency.
- Optional server-side LLM advisor for manual comparison when an API key is configured.
- Normalized deduplication and deterministic language×label-stratified holdout splitting.
- Five-fold comparison of Dummy, word TF-IDF, character TF-IDF, combined features, class weighting, and rating masking.
- Two calibrated, class-balanced ordinal Logistic Regression boundaries with validated decision thresholds and optional linear feature contributions.
- FastAPI `POST /classify` and `GET /health` endpoints.
- Repeated cold/warm component, service, explanation, and HTTP latency benchmarks.
- Pytest, coverage, Ruff, a non-root Dockerfile, MLflow tracking, and machine-readable model metadata.

## Architecture

```text
raw CSV (immutable)
  -> schema/hash audit
  -> local language identification
  -> normalized deduplication
  -> language + label stratified holdout and training CV
  -> [normalizer -> word/character TF-IDF -> two calibrated Logistic Regression boundaries]
  -> joblib model + JSON metadata
  -> FastAPI inference service
```

Language identification is separate from the sklearn pipeline so responses can report language and attach the English reliability warning. Normalization, vectorization, and classification stay in one shared fitted pipeline so Dutch and English use exactly the same model and training/serving cannot drift.

## Repository map

```text
configs/training.yaml             central experiment configuration
src/dutch_sentiment/              audit, data, language, model, API, benchmark code
tests/                            deterministic unit and API tests
artifacts/model.joblib            ready-to-serve fitted pipeline
artifacts/model_metadata.json     hashes, versions, split, metrics, schema
artifacts/*.json|*.csv            portable audit/experiment/benchmark evidence
reports/data_audit.md             interpreted full-dataset audit
reports/model_report.md           historical multiclass-baseline report
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
.venv/bin/python -m pip install -e '.[train,dev]'
```

Dependencies are constrained by compatible version ranges in `pyproject.toml`. The `train` extra contains MLflow/pandas/reporting tools; the Docker image installs only the smaller serving core. The verified environment resolved scikit-learn 1.9.0, Lingua 2.1.1, FastAPI 0.139.2, MLflow 3.14.0, and pytest 8.4.2.

## Exact working commands

```bash
make audit       # regenerate reports/data_audit.md and artifacts/data_audit.json
make train       # run CV experiments, MLflow tracking, select, fit, and evaluate once
make evaluate    # print stored held-out metrics; does not evaluate the test set again
make benchmark   # regenerate repeated latency evidence and refresh model report
make predict REVIEW='Deze film was verrassend goed.'
make test        # run pytest
make coverage    # run pytest with branch coverage
make lint        # Ruff lint and formatting checks
make serve       # listen on 0.0.0.0:8000
make mlflow      # open local MLflow UI on port 5000
```

## Train and predict: shortest workflow

Train from the supplied CSV and regenerate the fitted model, metadata, comparison table, and report:

```bash
make install
make audit
make train
make benchmark
```

Predict locally without starting a server:

```bash
.venv/bin/sentiment-predict \
  --model artifacts/model.joblib \
  --review 'Deze film was verrassend goed.' \
  --explain
```

The same operation is available through `make predict REVIEW='...'` or the REST API described below.

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
- Lingua candidates: 4,315 Dutch and 485 English. All are retained in one shared model. English contains 151 Positive, 324 Average, and only 10 Negative rows.
- Maximum source review length is 7,654 characters; the API limit is 8,000 and input is never silently truncated.

Detector output is not treated as annotated ground truth. The bounded manual samples and limitations are in `reports/data_audit.md`.

## Leakage-safe evaluation

Two same-label normalized duplicate extras are removed. With seed 42, language×label stratification places 3,838 rows in the training/CV partition and 960 in the originally held-out test partition. Training contains 3,450 Dutch and 388 English rows; test contains 863 Dutch and 97 English rows. Test label supports are Positive 450, Average 450, and Negative 60. Normalized review hashes prove the partitions are disjoint.

All normalization and TF-IDF fitting happens inside CV folds. The official TF-IDF workflow used the test set once after selecting the best mean CV macro-F1. Later isolated research branches reused the same fixed partition for comparisons, so it remains leakage-free with respect to model fitting but is not a new blind set for future promotion decisions.

## Experiment comparison

| Experiment | Features | Balanced | Ratings masked | CV macro-F1 mean ± std | CV balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| combined_balanced_ratings | word + char | yes | no | **0.6472 ± 0.0269** | **0.6220** |
| combined_balanced_masked_ratings | word + char | yes | yes | 0.6459 ± 0.0257 | 0.6207 |
| combined_logreg | word + char | no | no | 0.4922 ± 0.0140 | 0.4837 |
| char_logreg | char | no | no | 0.4721 ± 0.0066 | 0.4724 |
| word_logreg | word | no | no | 0.4474 ± 0.0105 | 0.4609 |
| dummy_prior | ignored baseline features | no | no | 0.2127 ± 0.0001 | 0.3333 |

Full metrics, standard deviations, MLflow run IDs, and artifact sizes are in `artifacts/experiment_comparison.csv`.

### Research branch index

The production submission remains on `main`. Completed or exploratory work stays isolated so optional dependencies and alternative serving designs do not blur the formal model contract.

| Branch | Scope | Headline result | Promotion status |
| --- | --- | --- | --- |
| [`experiment/linear-models`](https://github.com/ylceadap/sentiment-analysis/tree/experiment/linear-models) | Logistic Regression C sweep and LinearSVC | Best CV macro-F1 0.6536 | Frozen; predefined improvement gate not met |
| [`experiment/negative-imbalance`](https://github.com/ylceadap/sentiment-analysis/tree/experiment/negative-imbalance) | Custom class weights, Negative thresholds, and fold-local oversampling | Held-out Negative recall 0.65, precision 0.5909 | Frozen; precision gate of 0.60 not met |
| [`experiment/transformer-embeddings`](https://github.com/ylceadap/sentiment-analysis/tree/experiment/transformer-embeddings) | Frozen multilingual MiniLM and Dutch RobBERT sentence embeddings with Logistic Regression | Did not pass the OOF promotion gates | Frozen; official model unchanged |
| [`experiment/jina-embeddings`](https://github.com/ylceadap/sentiment-analysis/tree/experiment/jina-embeddings) | Frozen Jina v3 classification embeddings with Logistic Regression | Best OOF macro-F1 0.7108 | Research only; no held-out promotion evaluation and non-commercial model license |
| [`experiment/llm`](https://github.com/ylceadap/sentiment-analysis/tree/experiment/llm) | Direct DeepSeek V4 Flash few-shot classification | Held-out macro-F1 0.7506 | Separate architecture review required; external API, privacy, cost, latency, and repeated-test-set caveats |
| `experiment/ordinal-logistic` | Two calibrated cumulative Logistic Regression boundaries with monotonic composition and cross-fitted thresholds | Held-out macro-F1 0.6406; Negative recall 0.6167 | Promoted on this branch after passing the predefined replacement gate |

Only a candidate that wins on training-only CV/OOF evidence, is frozen, passes a new blind evaluation, satisfies deployment constraints, and passes CI should be proposed for merge into `main`.

### Ordinal logistic experiment

The ordinal experiment retains a classification objective while representing the natural label
order through two boundaries:

```text
Negative vs Average + Positive
Negative + Average vs Positive
```

Both boundaries reuse the submitted word/character TF-IDF features, balanced Logistic Regression,
and fold-internal three-fold sigmoid calibration. Independently estimated cumulative probabilities
are projected onto the monotonic constraint `P(y > Negative) >= P(y > Average)`. Thresholds are
evaluated with cross-fitting: the thresholds applied to each OOF fold are selected using only the
other OOF folds. `make ordinal-logistic` never evaluates the 960 held-out rows; the separate,
explicit `make promote-ordinal` command performs that comparison after the candidate is frozen.

Run the experiment with:

```bash
make ordinal-logistic
```

Detailed evidence is written to `artifacts/ordinal_logistic/ordinal_logistic_experiment.json` and
`artifacts/ordinal_logistic/ordinal_logistic_experiment.csv`. The selected training-only candidate
uses `C=1`, balanced boundary weights, sigmoid calibration, and full-OOF deployment thresholds
`0.75/0.55`.

| OOF metric | Multiclass baseline | Selected ordinal candidate | Change |
| --- | ---: | ---: | ---: |
| Macro-F1 | 0.6485 | **0.6529** | +0.0044 |
| Macro-F1 fold std | 0.0269 | **0.0154** | -0.0115 |
| Balanced accuracy | 0.6220 | **0.6497** | +0.0277 |
| Accuracy | 0.6699 | **0.6733** | +0.0034 |
| Negative precision | **0.7333** | 0.6256 | -0.1078 |
| Negative recall | 0.5042 | **0.5917** | +0.0875 |
| Negative F1 | 0.5975 | **0.6081** | +0.0106 |
| Ordinal MAE | 0.3468 | **0.3418** | -0.0050 |
| Quadratic weighted kappa | 0.4544 | **0.4755** | +0.0211 |
| Severe error rate | 0.0167 | **0.0151** | -0.0016 |

The candidate passed the predefined OOF gate and the subsequent held-out replacement gate. The
existing 960 rows were not fitted or used for threshold selection, but they have been inspected in
earlier repository experiments and therefore are no longer described as a new blind benchmark.
On this branch, the promoted ordinal artifact is now the model loaded by the CLI and API. A newly
collected, adjudicated blind set remains the right next step before making stronger generalization
claims.

### Rating-leakage result

Masking ratings reduced mean macro-F1 by 0.0014, much less than either experiment's fold-to-fold standard deviation. Retaining ratings therefore won the predefined selection metric, but the model does not appear heavily dependent on them. Ratings are legitimate review content and a plausible direct label cue; both interpretations remain documented.

## Selected model and held-out metrics

The selected model combines word unigrams/bigrams and character 3–5-grams with two class-balanced
Logistic Regression boundaries: `Negative` versus `Average + Positive`, and
`Negative + Average` versus `Positive`. Each boundary is sigmoid-calibrated; cumulative
probabilities are projected to satisfy the ordinal constraint, then frozen thresholds `0.75/0.55`
produce the label. It was promoted because it improved the predefined minority-class and aggregate
classification metrics while retaining CPU inference, serialization, probabilities, and
inspectable linear evidence.

| Metric | Held-out value |
| --- | ---: |
| Accuracy | 0.6500 |
| Balanced accuracy | 0.6404 |
| Macro precision | 0.6552 |
| Macro recall | 0.6404 |
| Macro-F1 | **0.6406** |
| Weighted F1 | 0.6461 |
| Log loss | 0.6889 |
| Multiclass Brier score | 0.4414 |
| Expected calibration error, 10 bins | 0.0342 |
| Mean prediction confidence | 0.6736 |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Positive | 0.7160 | 0.5378 | 0.6142 | 450 |
| Average | 0.6117 | 0.7667 | 0.6805 | 450 |
| Negative | **0.6379** | **0.6167** | **0.6271** | 60 |

Negative is the smallest class; 37/60 held-out Negative reviews were correctly classified, 19
became Average, and 4 became Positive. Compared with the previous multiclass artifact, accuracy fell
from 0.6531 to 0.6500 and ordinal MAE moved from 0.36146 to 0.36250 (one additional total ordinal
error point across 960 rows), while severe Positive/Negative errors fell from 14 to 12 and quadratic
weighted kappa rose from 0.4388 to 0.4517. The machine-readable comparison is in
`artifacts/ordinal_logistic/ordinal_logistic_held_out_evaluation.json`.

The probability metrics are descriptive evidence on 960 held-out rows. Both ordinal boundaries use
training-only sigmoid calibration, but ECE on a reused evaluation set is not an operational
guarantee.

### One model, evaluated by language

| Detected language | Support | Accuracy | Balanced accuracy | Macro-F1 | Negative F1 / support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dutch | 863 | 0.6466 | 0.6463 | **0.6437** | 0.6379 / 58 |
| English | 97 | 0.6804 | 0.3444 | **0.2907** | 0.0000 / 2 |

These are evaluation slices of the same model, not separately trained models. English accuracy is
dominated by 65 Average examples: the ordinal model predicted all 65 correctly, but only 1/30
English Positive and 0/2 English Negative examples correctly. This is why every English prediction
carries a reliability warning.

## Latency evidence

Measured on macOS x86_64, Python 3.11.7, using three short/medium/long Dutch samples, 10 warm-ups, and 100 iterations (HTTP: 50):

| Path | Mean ms | p50 ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| Normalization + model | 3.906 | 3.819 | 6.288 | 6.569 |
| Language detection + model | 5.143 | 4.835 | 8.174 | 8.945 |
| Service end-to-end | 5.666 | 5.320 | 8.807 | 9.711 |
| Explanation enabled | 6.499 | 5.950 | 9.477 | 10.684 |
| HTTP `/classify` | 8.045 | 7.571 | 11.146 | 11.778 |

- Lingua constructor: 0.369 ms, but its lazy first inference/model load: 1,610.091 ms.
- Serialized model load: 630.598 ms; first model prediction: 12.543 ms.
- Artifact: 2,935,649 bytes; SHA-256 `32ec6bc66d70c26f50bc7b6f495d0852cdd1ee0fd68cbff97d823b34370bf836`.

The shared-language model remains within the same small CPU-serving envelope. The API warms Lingua and the explanation feature-name cache during lifespan startup, so readiness includes the cold initialization cost instead of passing it to the first request.

## REST API

Start the API:

```bash
make serve
```

The default browser entry point is a small interactive app for manual inference:

```text
http://localhost:8000/
```

The browser app compares the submitted local model with an optional advisory LLM
recommendation. The local model is always available and remains the formal output. The LLM
panel is disabled unless the server can load a key from an environment variable or an ignored
local key file.

Interactive OpenAPI documentation, including English field descriptions and runnable request/response examples, is available at:

```text
http://localhost:8000/docs
```

Health:

```bash
curl -sS http://localhost:8000/health
```

```json
{"status":"ok","model_version":"0.1.0+...","model_ready":true}
```

Classify one Dutch or English review:

```bash
curl -sS -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"review":"Deze film was verrassend goed, met sterke acteurs en een mooi einde.","explain":false}'
```

The response contains one exact allowed label plus the composed ordinal probabilities:

```json
{
  "label": "Average",
  "model_version": "0.1.0+ordinal.0d0f0f84",
  "detected_language": "dutch",
  "probabilities": {"Average": 0.6413132, "Negative": 0.0404957, "Positive": 0.3181911},
  "latency_ms": 5.32,
  "warnings": [],
  "explanation": null
}
```

Set `"explain": true` to receive supporting/opposing word n-grams plus separately labeled technical character n-grams. These are linear contributions, not causal explanations.

Compare the submitted model with the optional LLM advisor:

```bash
curl -sS -X POST http://localhost:8000/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"review":"Deze film was verrassend goed, met sterke acteurs en een mooi einde.","explain":false}'
```

Enable the LLM panel for local manual use by putting the key in the ignored local secrets file:

```bash
mkdir -p .secrets
printf '%s\n' 'your-deepseek-key' > .secrets/deepseek_api_key
make serve
```

The LLM advisor uses an OpenAI-compatible chat completions call. Optional overrides are
`DEEPSEEK_API_KEY`, `LLM_API_KEY`, `DEEPSEEK_API_KEY_FILE`, `LLM_API_KEY_FILE`,
`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_TIMEOUT_SECONDS`.

English input is accepted by the same model and carries an explicit warning:

```json
{
  "label": "Average",
  "model_version": "0.1.0+9829f359",
  "detected_language": "english",
  "probabilities": {"Average": 0.4985972, "Negative": 0.1243259, "Positive": 0.3770768},
  "latency_ms": 16.38,
  "warnings": [
    "English predictions are supported, but less reliable because English training data is limited and highly class-imbalanced."
  ],
  "explanation": null
}
```

### Validation and language policy

- Missing, blank, whitespace-only, extra-field, or longer-than-8,000-character input returns HTTP 422.
- Confident Dutch and English input is accepted. Confidently identified languages outside that training scope return HTTP 422:

```json
{"detail":"The review was confidently detected as an unsupported language; submit a Dutch or English review."}
```

- Text shorter than 20 characters is treated as internally ambiguous and allowed through instead of being falsely rejected. The response still returns the detector's top `detected_language`; an English top candidate receives the same reliability warning. This trades some unsupported-language false acceptance for safer behavior on valid short input such as “Goed”.
- Unexpected prediction failures return a generic HTTP 500 without the internal exception.
- Logs include request ID, length, detected language, label, latency, and error category—not full review text.

## Tests and quality

Verified commands:

```bash
make test
# 26 passed, 1 third-party Starlette deprecation warning

make coverage
# 26 tests passed; 58% total branch coverage
# Critical API, audit, data, language, metrics, model, service, and text logic is directly tested

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

Then repeat the health and classify curl commands. The image uses Python 3.11 slim, installs only the core serving dependencies (not MLflow, pandas, pyarrow, matplotlib, or training tools), runs as a non-root user, includes a health check, and copies the model/metadata but excludes the raw CSV, PDF, tests, reports, Git data, and MLflow database.

**Verification status:** the local development environment has no Docker engine, but GitHub Actions successfully built the image, started the container, and reached `/health` on 2026-07-18 ([CI run 1](https://github.com/ylceadap/sentiment-analysis/actions/runs/29649585086)).

## Traceability and security

- Git commits track implementation phases.
- Raw data, split content, fitted model, and package versions are hashed/recorded.
- `artifacts/model_metadata.json` includes the training timestamp, Git commit, language config, label classes, schema, split evidence, thresholds, promotion checks, previous-model evidence, and held-out metrics.
- The original CSV remains unchanged and is rehashed during verification.
- `joblib`/pickle artifacts can execute code while loading. Only load the repository's locally controlled artifact; never accept an untrusted uploaded model.
- The submitted local model does not require secrets or hosted services. The optional LLM advisor
  reads a server-side key from environment variables or the ignored `.secrets/deepseek_api_key`
  file and should be used only for manual comparison.

## Limitations and sensible next steps

- Lingua can misclassify mixed, translated, short, or named-entity-heavy text.
- Sparse n-grams do not deeply understand sarcasm, negation scope, or long-range composition.
- English has only 485 raw rows, including 10 Negative rows; its held-out macro-F1 is 0.2907 and English Negative support is only 2.
- Negative remains scarce overall: 300 raw and 60 held-out rows.
- Source labels and their possible derivation from ratings were not independently verified.
- There are no timestamps/source IDs for drift or lineage analysis.
- Startup capacity and readiness time should account for Lingua's approximately 1.6-second cold loading cost.
- A compact multilingual transformer is a future experiment only after collecting a larger adjudicated benchmark and defining latency/cost constraints.
