# Requirements Traceability

The challenge PDF is the authoritative source. `Required` and `Bonus` below reflect its wording; additional safeguards come from the execution brief.

| Requirement | Priority | Planned implementation | Verification/evidence | Status |
| --- | --- | --- | --- | --- |
| Consider Dutch-language reviews only | Required | Local deterministic language component used during training and inference | 2 language tests pass; `reports/data_audit.md`; API non-Dutch test pending | In progress |
| Return exactly Positive, Average, or Negative | Required | Typed label contract in model and API | Model/API tests; OpenAPI schema | Pending |
| Create useful features | Required | Compared word and character TF-IDF feature sets; optional small engineered features only with evidence | MLflow runs; comparison table; model report | Pending |
| Experiments easy to revisit and compare | Required | Local MLflow tracking plus exported comparison CSV/Markdown and configs | MLflow UI command; run artifacts; model report | Pending |
| Proper software design and OOP | Required | Focused loader/normalizer/language/model/service boundaries in a `src` package | Source review; unit tests | Pending |
| At least one unit test for a class, preferably model class | Required | Model fit/predict and save/load round-trip tests | `pytest` output | Pending |
| REST POST `/classify` for one review | Required | FastAPI application factory and typed request/response | API tests; live curl check | Pending |
| Response label is one of three exact labels | Required | Enum/literal validation and model contract | Unit and API tests | Pending |
| Inference latency considered | Required | Cold/warm component and end-to-end benchmark with p50/p95 | Benchmark JSON/table; model report | Pending |
| Prediction explanation | Bonus | Optional local linear feature contributions with source labels | Explanation tests and example | Pending |
| Data/model versioning | Bonus | Git, raw/model SHA-256, MLflow run, config, split hashes, versioned metadata | Git log; `model_metadata.json`; reports | In progress |
| Serve in Docker | Bonus | Non-root API image excluding the raw dataset | Dockerfile review; build/run checks if possible | Blocked for runtime verification: Docker unavailable |
| Small usable app and run instructions | Required | FastAPI service, task commands, evaluator-focused README | Executed README commands | Pending |

## Expanded engineering safeguards

| Safeguard | Implementation/evidence | Status |
| --- | --- | --- |
| Preserve source CSV | Read-only input; recheck SHA-256 `2788b987e2c9fa4fd6459a1798a6e7d1dd63ddb5618c10907712d39042cc4be2` | Verified initially |
| Avoid ordered-split and duplicate leakage | Shuffled stratified grouped/deduplicated holdout and CV | Pending |
| Investigate explicit-rating leakage | Matched retained-versus-masked experiment | Pending |
| Report imbalance-aware metrics | Macro-F1, balanced accuracy, per-class metrics, confusion matrix | Pending |
| Keep training and inference preprocessing aligned | Serialized sklearn pipeline plus shared language policy | Pending |
| Validate API inputs and privacy-safe logging | Blank/length/language checks; metadata-only logs | Pending |
| Produce reproducible data audit | CLI-generated `reports/data_audit.md`; all-row JSON evidence | Complete |
| Record actual verification only | Commands/results and environment limitations in reports/README | In progress |
