# Requirements Traceability

The challenge PDF is the authoritative source. `Required` and `Bonus` below reflect its wording; additional safeguards come from the execution brief.

| Requirement | Priority | Planned implementation | Verification/evidence | Status |
| --- | --- | --- | --- | --- |
| Consider Dutch-language reviews only | Required | Local deterministic language component used during training and inference | 2 language tests pass; `reports/data_audit.md`; API non-Dutch test pending | In progress |
| Return exactly Positive, Average, or Negative | Required | Typed label contract in model and API | Model/API tests; OpenAPI schema | Complete |
| Create useful features | Required | Compared word and character TF-IDF feature sets | MLflow runs; comparison table; model report | Complete |
| Experiments easy to revisit and compare | Required | Local SQLite MLflow plus exported comparison CSV/Markdown and configs | MLflow UI command; run IDs; model report | Complete |
| Proper software design and OOP | Required | Focused loader/normalizer/language/model/service boundaries in a `src` package | Source review; direct unit tests | Complete |
| At least one unit test for a class, preferably model class | Required | Model fit/predict, explanation, probability, deterministic and save/load tests | 23 passing tests | Complete |
| REST POST `/classify` for one review | Required | FastAPI application factory and typed request/response | API tests; live curl 200 | Complete |
| Response label is one of three exact labels | Required | Literal validation and model contract | Unit and API tests | Complete |
| Inference latency considered | Required | Cold/warm component and end-to-end benchmark with p50/p95 | `benchmark.json`; model report | Complete |
| Prediction explanation | Bonus | Optional local linear feature contributions with source labels | Model test and live API example | Complete |
| Data/model versioning | Bonus | Git, raw/model SHA-256, MLflow run, config, split hashes, versioned metadata | Git log; `model_metadata.json`; reports | Complete |
| Serve in Docker | Bonus | Non-root API image excluding the raw dataset | Dockerfile review; build/run checks if possible | Blocked for runtime verification: Docker unavailable |
| Small usable app and run instructions | Required | FastAPI service, task commands, evaluator-focused README | Live health/classify checks; README | Complete |

## Expanded engineering safeguards

| Safeguard | Implementation/evidence | Status |
| --- | --- | --- |
| Preserve source CSV | Read-only input; recheck SHA-256 `2788b987e2c9fa4fd6459a1798a6e7d1dd63ddb5618c10907712d39042cc4be2` | Verified initially |
| Avoid ordered-split and duplicate leakage | Shuffled stratified deduplicated holdout and CV; disjoint hashes | Complete |
| Investigate explicit-rating leakage | Matched retained-versus-masked experiment | Complete |
| Report imbalance-aware metrics | Macro-F1, balanced accuracy, per-class metrics, confusion matrix | Complete |
| Keep training and inference preprocessing aligned | Serialized sklearn pipeline plus shared language policy | Complete |
| Validate API inputs and privacy-safe logging | Blank/length/language tests; metadata-only logs | Complete |
| Produce reproducible data audit | CLI-generated `reports/data_audit.md`; all-row JSON evidence | Complete |
| Record actual verification only | Commands/results and environment limitations in reports/README | Complete |
