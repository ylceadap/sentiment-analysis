# Implementation Plan

This plan is maintained as verification evidence. A phase is complete only when its listed checks have actually run successfully; environment limitations are recorded explicitly.

## Phase 0 — Requirements and workspace inspection

**Objective:** establish the authoritative requirements, inputs, constraints, and risks before application implementation.

- [x] Read the complete challenge PDF.
- [x] Inspect all initial workspace files.
- [x] Verify the raw CSV hash and basic file characteristics.
- [x] Confirm repository and tool availability.
- [x] Create the initial plan, decision log, and traceability matrix.

**Validation commands**

```bash
pdfinfo Python_Engineer_coding_challenge_1_1.pdf
pdftotext -layout Python_Engineer_coding_challenge_1_1.pdf -
shasum -a 256 Python_Engineer_Challenge_2.csv
file Python_Engineer_Challenge_2.csv
```

**Status:** complete.

**Unresolved risks at this phase:** Docker was not installed locally. Runtime verification was later completed in GitHub Actions after the repository was published.

## Phase 1 — Reproducible data audit

**Objective:** establish whether the CSV is safe for modeling and define cleaning, language, deduplication, and split policies.

- [x] Implement a deterministic audit command.
- [x] Profile schema, missingness, labels, ordering, duplicates, lengths, markup, invisible Unicode, mojibake candidates, and ratings.
- [x] Inspect representative and problematic samples without reproducing excessive review text.
- [x] Add local deterministic language-identification results and review samples manually.
- [x] Write `reports/data_audit.md` with evidence, risks, and remediation.
- [x] Verify the source CSV remains byte-identical.

**Validation commands**

```bash
python -m dutch_sentiment.audit --data Python_Engineer_Challenge_2.csv --output reports/data_audit.md
shasum -a 256 Python_Engineer_Challenge_2.csv
```

**Status:** complete. Evidence: 11 focused data/text/language tests passed; Ruff lint passed; report generated from all 4,800 rows.

**Unresolved risks:** automated language identification is imperfect for short or mixed-language reviews; the report will distinguish detector output from ground truth.

## Phase 2 — Project structure and deterministic data pipeline

**Objective:** create a small, installable `src`-layout package with shared training/inference normalization and leakage-safe data preparation.

- [x] Add constrained runtime/development dependencies, initial CLI entry points, logging, and configuration.
- [x] Implement and test conservative text normalization.
- [x] Implement and test a deterministic Dutch-language policy.
- [x] Implement grouped/deduplicated stratified holdout splitting.
- [x] Persist split metadata and relevant hashes.

**Validation commands**

```bash
python -m pip install -e '.[dev]'
pytest tests/test_data.py tests/test_text.py tests/test_language.py
ruff check .
ruff format --check .
```

**Status:** superseded by Phase 7. The original Dutch-only split had 3,450/863 rows; the user later approved one shared Dutch/English model using all supplied rows.

**Unresolved risks:** dependency installation and detector model availability must be verified locally.

## Phase 3 — Reproducible experiments and model selection

**Objective:** compare transparent classical NLP baselines without touching the held-out test set during selection.

- [x] Run Dummy, word TF-IDF, character TF-IDF, and combined word/character TF-IDF experiments.
- [x] Compare imbalance treatment with stratified CV using macro-F1 as the primary selection metric.
- [x] Compare otherwise-equivalent retained-rating and masked-rating experiments.
- [x] Track meaningful experiments in local MLflow and export a compact comparison table.
- [x] Select the final candidate from CV evidence, then evaluate once on the held-out test set.
- [x] Perform concise error analysis and create `reports/model_report.md`.
- [x] Serialize the fitted pipeline and machine-readable traceability metadata.

**Validation commands**

```bash
python -m dutch_sentiment.train --config configs/training.yaml
python -m dutch_sentiment.evaluate --model artifacts/model.joblib
mlflow ui --backend-store-uri ./mlruns
```

**Status:** superseded by Phase 7 retraining. The final unified model has CV macro-F1 0.6472 ± 0.0269, held-out macro-F1 0.6379, and Negative F1 0.6019.

**Unresolved risks:** Negative has only 300 raw rows, so per-class estimates may have material variance; selection will report fold dispersion and Negative support.

## Phase 4 — Inference API, explanations, and latency

**Objective:** expose the immutable fitted pipeline safely through a testable FastAPI application.

- [x] Implement a model abstraction with save/load and linear feature contributions.
- [x] Implement application factory/lifespan loading, `/classify`, and `/health`.
- [x] Reject missing, blank, oversized, and confidently unsupported-language inputs clearly.
- [x] Keep explanations optional and avoid logging review text.
- [x] Benchmark initialization, loading, warm inference, language-plus-model, explanation, and HTTP paths.

**Validation commands**

```bash
python -m dutch_sentiment.benchmark --model artifacts/model.joblib
uvicorn dutch_sentiment.api:create_app --factory --host 0.0.0.0 --port 8000
curl -fsS http://localhost:8000/health
```

**Status:** superseded by Phase 7 API policy. Dutch and English now return 200; English includes a reliability warning, while other confidently identified languages return 422.

**Unresolved risks:** explanation readability depends on the selected feature union; word features will be presented separately from technical character features.

## Phase 5 — Tests, container, documentation, and acceptance

**Objective:** leave a submission that an evaluator can install, inspect, verify, and serve using documented commands.

- [x] Add deterministic unit, model round-trip, explanation, and API tests.
- [x] Run tests, coverage, lint, and formatting checks; record actual results.
- [x] Add a non-root serving Dockerfile that excludes the raw CSV.
- [x] Check Docker availability locally, then verify image build, container startup, and `/health` through GitHub Actions.
- [x] Complete README, model report, decision log, and requirement traceability.
- [x] Reread the PDF and run the final acceptance checklist.
- [x] Recheck the raw CSV hash.
- [x] Commit each verified phase with a focused message.

**Validation commands**

```bash
pytest
pytest --cov=dutch_sentiment --cov-report=term-missing
ruff check .
ruff format --check .
docker build -t dutch-sentiment .
docker run --rm -p 8000:8000 dutch-sentiment
```

**Status:** complete. Local evidence recorded 24 passing tests, 58% total branch coverage with high coverage of critical logic, passing Ruff checks, a live application lifecycle, and reconciled source/model/metadata hashes. GitHub Actions later verified the Docker image build and runtime health check.

**Unresolved risks:** Docker remains unavailable locally, but the Linux CI runtime check is complete.

## Phase 6 — Post-completion performance and usability review

**Objective:** remove avoidable serving work, strengthen probability evidence, reduce container dependencies, and make one-off prediction easier.

- [x] Replace repeated predict/probability/explanation transformations with one sparse-vector inference operation.
- [x] Cache feature names after first explanation and warm the cache during API startup.
- [x] Add held-out log loss, multiclass Brier score, 10-bin expected calibration error, and mean confidence.
- [x] Replace warning-prone CV metric strings with zero-division-safe scorers.
- [x] Separate core serving dependencies from `train` and `dev` extras.
- [x] Remove unused API YAML configuration and share the maximum-input constant.
- [x] Add a local `sentiment-predict` command in addition to REST inference.
- [x] Retrain from a clean Git commit and rerun latency benchmarks.

**Validation commands**

```bash
.venv/bin/pytest --cov=dutch_sentiment --cov-report=term-missing
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/sentiment-predict --review 'Deze film was verrassend goed.' --explain
.venv/bin/sentiment-benchmark --model artifacts/model.joblib
```

**Status:** complete. Classification metrics remained reproducible. Service p50 improved from 6.778 to 5.752 ms, HTTP p50 from 8.292 to 7.780 ms, and explanation p50 from 131.707 to 7.549 ms.

**Remaining risks:** the probability metrics are held-out descriptive estimates rather than a separately calibrated deployment guarantee. Docker is unavailable locally, while Linux container execution is verified in CI.

**Final verification:** 24 tests passed, total branch coverage is 58%, and both Ruff lint and format checks passed.

## Phase 7 — Unified Dutch/English training and transparent reliability warning

**Objective:** use all supplied Dutch and English reviews in one model while making the weaker English evidence explicit.

- [x] Replace Dutch filtering with row-level language annotation that retains all supplied rows.
- [x] Deduplicate before one unified language×label-stratified holdout split.
- [x] Use the same language×label stratification across model-selection CV folds.
- [x] Train one shared model rather than separate Dutch and English models.
- [x] Add overall and per-language held-out metrics.
- [x] Accept English in local/HTTP inference and return `detected_language` plus an explicit warning.
- [x] Keep confidently identified languages outside Dutch/English unsupported.
- [x] Retrain from clean Git commit `9829f35` and regenerate the model report and latency benchmark.

**Measured evidence:** 4,798 deduplicated rows; 3,838 train and 960 test. Overall held-out macro-F1 is 0.6379. Dutch macro-F1 is 0.6381 over 863 rows. English macro-F1 is 0.3289 over 97 rows; English Negative has only 2 held-out examples and F1 0, so the warning is mandatory.

**Remaining risk:** the user's all-sample scope intentionally extends the challenge's literal Dutch-only wording. English performance is descriptive and strongly biased toward Average; it must not be presented as production-grade bilingual support.

**Final verification:** 26 tests passed, total branch coverage is 58%, Ruff lint/format checks passed, and real Dutch/English/unsupported-language API paths were exercised against the serialized model.
