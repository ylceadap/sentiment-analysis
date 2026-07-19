# MLflow Model Registry

The local MLflow store separates deployable model artifacts from experiment evidence. A high metric
alone does not make an experiment a production model.

## Registered model policy

| Registered model | Alias | Tier | Deployment status |
| --- | --- | --- | --- |
| `sentiment-production` | `champion` | Production | Active local service model |
| `sentiment-dummy-prior` | `baseline` | Baseline | Evaluation only |
| `sentiment-word-logreg` | `benchmark` | Benchmark | Frozen |
| `sentiment-char-logreg` | `benchmark` | Benchmark | Frozen |
| `sentiment-combined-logreg` | `benchmark` | Benchmark | Frozen |
| `sentiment-combined-balanced` | `benchmark` | Benchmark | Selected training candidate; final fit is `sentiment-production` |
| `sentiment-combined-balanced-masked` | `benchmark` | Ablation | Frozen |
| `sentiment-linear-svc` | `frozen-challenger` | Challenger | Promotion gate failed |
| `sentiment-frozen-robbert-embeddings` | `research-only` | Research | Not self-contained; promotion gates failed |
| `sentiment-deepseek-v4-flash-24shot` | `external-advisor` | External API | No model weights stored; architecture review required |
| `sentiment-jina-v3-logreg` | `research-only` | Research | Evidence-only registry entry; no blind test; Jina v3 is non-commercial |
| `sentiment-jina-v3-ordinal-logistic` | `research-only` | Research | Strongest local OOF evidence; no blind test; Jina v3 is non-commercial |
| `sentiment-tfidf-ordinal-logistic` | `frozen-challenger` | Challenger | Branch artifact; needs a new blind test before production promotion |

Only `sentiment-production@champion` is approved for the submitted service. Aliases on separate
registered-model names are descriptive; the governance tags are the authoritative eligibility
record.

Every registered model also has one `artifact.tier`:

- `deployable`: a self-contained fitted artifact exists, even when governance forbids promotion;
- `reproducible`: code/configuration and external revisions exist but the artifact is not standalone;
- `evidence-only`: metrics and reports exist without locally stored model weights.

## Experiment evidence

The `dutch-sentiment-research-evidence` experiment contains one immutable catalog run per completed
branch. These runs store metrics, configurations, decision JSON/CSV files, reports, the source branch,
and the exact Git commit. They are tagged `evidence_only=true` and are not presented as registered
deployable models.

The catalog covers linear models, Negative-class imbalance, frozen transformer embeddings, Jina v3
embeddings, DeepSeek classification, ordinal regression, ordinal logistic, and Jina ordinal logistic.
The top Jina entries are now also visible in the registry as research-only records, not deployable
champions. The ordinal-logistic evidence run also retains its branch model artifact, but it remains a
challenger because the reused holdout is not a new blind benchmark. Jina results remain research-only
because no blind evaluation was run and Jina Embeddings v3 uses a non-commercial license.
That license is acceptable for this non-commercial assignment research and remains recorded in tags.

`sentiment-deepseek-v4-flash-24shot` is historical 24-shot experiment evidence. The production UI's
optional advisor uses the explicitly versioned `zero-shot-advisor-v1` prompt, so it is not presented
as the same evaluated configuration and has no Registry deployment authority.

## Reapply or audit

Run the idempotent policy script from the repository root:

```bash
.venv/bin/python scripts/organize_mlflow_registry.py
.venv/bin/python scripts/organize_mlflow_registry.py --audit-only
.venv/bin/python scripts/manage_model_release.py verify --require-mlflow
```

Open the local UI with `make mlflow`. The SQLite database and `mlruns/` artifacts are intentionally
ignored by Git, so they require a separate backup before the workspace is removed.
