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

**Unresolved risks:** Docker is not installed in the current environment; runtime container verification may remain unavailable. Git is being initialized at the user's request, so commit provenance will be captured after phase validation.

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
- [ ] Implement grouped/deduplicated stratified holdout splitting.
- [ ] Persist split metadata and relevant hashes.

**Validation commands**

```bash
python -m pip install -e '.[dev]'
pytest tests/test_data.py tests/test_text.py tests/test_language.py
ruff check .
ruff format --check .
```

**Status:** pending.

**Unresolved risks:** dependency installation and detector model availability must be verified locally.

## Phase 3 — Reproducible experiments and model selection

**Objective:** compare transparent classical NLP baselines without touching the held-out test set during selection.

- [ ] Run Dummy, word TF-IDF, character TF-IDF, and combined word/character TF-IDF experiments.
- [ ] Compare imbalance treatment with stratified CV using macro-F1 as the primary selection metric.
- [ ] Compare otherwise-equivalent retained-rating and masked-rating experiments.
- [ ] Track meaningful experiments in local MLflow and export a compact comparison table.
- [ ] Select the final candidate from CV evidence, then evaluate once on the held-out test set.
- [ ] Perform concise error analysis and create `reports/model_report.md`.
- [ ] Serialize the fitted pipeline and machine-readable traceability metadata.

**Validation commands**

```bash
python -m dutch_sentiment.train --config configs/training.yaml
python -m dutch_sentiment.evaluate --model artifacts/model.joblib
mlflow ui --backend-store-uri ./mlruns
```

**Status:** pending.

**Unresolved risks:** Negative has only 300 raw rows, so per-class estimates may have material variance; selection will report fold dispersion and Negative support.

## Phase 4 — Inference API, explanations, and latency

**Objective:** expose the immutable fitted pipeline safely through a testable FastAPI application.

- [ ] Implement a model abstraction with save/load and linear feature contributions.
- [ ] Implement application factory/lifespan loading, `/classify`, and `/health`.
- [ ] Reject missing, blank, oversized, and confidently non-Dutch inputs clearly.
- [ ] Keep explanations optional and avoid logging review text.
- [ ] Benchmark initialization, loading, warm inference, language-plus-model, explanation, and HTTP paths.

**Validation commands**

```bash
python -m dutch_sentiment.benchmark --model artifacts/model.joblib
uvicorn dutch_sentiment.api:create_app --factory --host 0.0.0.0 --port 8000
curl -fsS http://localhost:8000/health
```

**Status:** pending.

**Unresolved risks:** explanation readability depends on the selected feature union; word features will be presented separately from technical character features.

## Phase 5 — Tests, container, documentation, and acceptance

**Objective:** leave a submission that an evaluator can install, inspect, verify, and serve using documented commands.

- [ ] Add deterministic unit, model round-trip, explanation, and API tests.
- [ ] Run tests, coverage, lint, and formatting checks; record actual results.
- [ ] Add a non-root serving Dockerfile that excludes the raw CSV.
- [ ] Build/run/test the image if Docker becomes available; otherwise perform static review and mark runtime verification unavailable.
- [ ] Complete README, model report, decision log, and requirement traceability.
- [ ] Reread the PDF and run the final acceptance checklist.
- [ ] Recheck the raw CSV hash.
- [ ] Commit each verified phase with a focused message.

**Validation commands**

```bash
pytest
pytest --cov=dutch_sentiment --cov-report=term-missing
ruff check .
ruff format --check .
docker build -t dutch-sentiment .
docker run --rm -p 8000:8000 dutch-sentiment
```

**Status:** pending.

**Unresolved risks:** Docker runtime checks are currently blocked because the executable is unavailable.
