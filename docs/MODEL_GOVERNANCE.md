# Model Governance and Evidence

Git stores reproducible source and portable experiment summaries. MLflow stores run parameters,
metrics, artifacts, and lifecycle metadata. A strong score does not by itself authorize deployment.

## Deployment boundary

- `sentiment-production@champion` is the only model approved for the submitted API and local UI.
- Challenger aliases identify frozen candidates; they do not authorize deployment.
- Jina models remain research-only because the pinned encoder is CC-BY-NC-4.0.
- The LLM few-shot result is external-advisor evidence; provider weights are not stored locally.
- RobBERT weights remain in their verified Google Drive bundles and are not duplicated in Git or the
  local Registry artifact store.
- Exactly seven records are selected for presentation. All other registered models are benchmark,
  ablation, baseline, or test-only evidence.

## Final presentation models

| Rank | Registered model | Governance role | Deployment status |
| ---: | --- | --- | --- |
| 1 | `sentiment-deepseek-v4-flash-24shot` | External advisor | Historical API evidence only |
| 2 | `sentiment-jina-v3-ordinal-logistic` | Research only | Non-commercial encoder |
| 3 | `sentiment-jina-v3-logreg` | Research only | Non-commercial encoder |
| 4 | `sentiment-robbert-v2-improved` | Challenger evaluation | GPU-oriented, weights in Drive |
| 5 | `sentiment-tfidf-ordinal-logistic` | Frozen challenger | Not promoted |
| 6 | `sentiment-production` | Production benchmark | Active `champion` |
| 7 | `sentiment-robbert-v2-logistic` | Presentation test evidence | Not promoted |

The unified result is stored in MLflow experiment `dutch-sentiment-final-comparison`, run
`688b28b059dd477693b87104c32fbb9a`, and in `artifacts/final_models/`. The ranking uses the same
960-row test split for every frozen candidate. It is a reused-heldout presentation comparison, not
a new blind promotion test.

Each Registry record also has an artifact tier:

- `deployable`: a self-contained fitted artifact exists, although governance may still forbid use;
- `reproducible`: code, configuration, and pinned external revisions exist;
- `evidence-only`: metrics and reports exist without locally stored weights.

## Archived Git and MLflow evidence

Completed experiment branches are replaced by immutable remote tags. `main` is the only long-lived
branch.

| Experiment family | Archive tag | Source SHA | MLflow evidence run |
| --- | --- | --- | --- |
| Linear models | `archive/linear-models/2026-07-19` | `236d189` | `cc28bdf35d174d67bb18f58c64b6bb21` |
| Negative imbalance | `archive/negative-imbalance/2026-07-19` | `b2cba5d` | `4508ae4d5d8d4e25a519c97ec2368c03` |
| Frozen transformer embeddings | `archive/transformer-embeddings/2026-07-19` | `88854ad` | `8ae4bc35d9c04c95982a8454f13dc545` |
| Jina embeddings | `archive/jina-embeddings/2026-07-19` | `52a9a32` | `67e40778b5b04f2b8728a5626794fd3b` |
| LLM advisor | `archive/llm/2026-07-19` | `5841ae9` | `a3ffde34a7634f51b47dc270ec4f710e` |
| Ordinal regression | `archive/ordinal-regression/2026-07-19` | `eafcf64` | `8ab158b7016c4b30b81ae872e00a40f6` |
| TF-IDF ordinal logistic | `archive/ordinal-logistic/2026-07-19` | `7503f9e` | `9db0c255fb9f4c46b7fb7654953d934c` |
| Jina ordinal logistic | `archive/jina-ordinal-logistic/2026-07-19` | `f52ad65` | `c030142c6d3e46f78beb730c680a54e8` |
| Alternate Jina ordinal lineage | `archive/jina-ordinal-from-ordinal/2026-07-19` | `d1e14fb` | Covered by Jina/ordinal evidence |

The alternate lineage is retained only for ancestry recovery. It is not a separate trained model.
A branch may be deleted only after its archive tag is available on `origin` and the associated
MLflow evidence has been verified.

## RobBERT evidence

The paired RobBERT run is stored in MLflow experiment
`dutch-sentiment-robbert-v2-finetuning`, run `74b1a804f3694d66a0a8c7c9360c1e6d`.
Logistic is selected for presentation; CORAL remains test-only.

The improvement sweep is stored in `dutch-sentiment-robbert-improvement`, run
`62a5a68d9e1d48c6832b94641524f53e`. Its head-tail 512 weighted ensemble reached test Macro-F1
0.6615 and remains a challenger. Full experiment details are in `reports/robbert_experiments.md`.

## External advisor

The optional UI advisor reuses the frozen `deterministic-24-shot-v1` prompt associated with the
historical LLM result. Its prompt SHA-256 is
`d4ca19fd4f4bb457a5d36ed4e90e1bcf925157a7f7f36496d3c8ab0c1fa0b908`. Live provider behavior may
change and has no authority over the TF-IDF champion.

## Reapply and audit

```bash
.venv/bin/python scripts/organize_mlflow_registry.py
.venv/bin/python scripts/organize_mlflow_registry.py --audit-only
.venv/bin/python scripts/manage_model_release.py verify --require-mlflow
```

The SQLite database and `mlruns/` are intentionally ignored by Git. They require an independent
backup before the workspace is removed.
