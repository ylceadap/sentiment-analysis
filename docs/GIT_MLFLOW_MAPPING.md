# Git and MLflow evidence map

Git stores reproducible code and immutable experiment summaries. MLflow stores run parameters,
metrics, model artifacts, and lifecycle metadata. The archive tags below replace long-lived
experiment branches as the permanent source-code references.

## Archived experiment evidence

| Experiment family | Archive tag | Source SHA | MLflow catalog ID | Evidence run ID | Registry record |
| --- | --- | --- | --- | --- | --- |
| Linear models | `archive/linear-models/2026-07-19` | `236d189` | `linear-models-v1` | `cc28bdf35d174d67bb18f58c64b6bb21` | `sentiment-linear-svc` |
| Negative-class imbalance | `archive/negative-imbalance/2026-07-19` | `b2cba5d` | `negative-imbalance-v1` | `4508ae4d5d8d4e25a519c97ec2368c03` | Production-family benchmark evidence |
| Frozen transformer embeddings | `archive/transformer-embeddings/2026-07-19` | `88854ad` | `transformer-embeddings-v1` | `8ae4bc35d9c04c95982a8454f13dc545` | `sentiment-frozen-robbert-embeddings` |
| Jina embeddings | `archive/jina-embeddings/2026-07-19` | `52a9a32` | `jina-embeddings-v1` | `67e40778b5b04f2b8728a5626794fd3b` | `sentiment-jina-v3-logreg` |
| DeepSeek advisor | `archive/llm/2026-07-19` | `5841ae9` | `llm-deepseek-v1` | `a3ffde34a7634f51b47dc270ec4f710e` | `sentiment-deepseek-v4-flash-24shot` |
| Ordinal regression | `archive/ordinal-regression/2026-07-19` | `eafcf64` | `ordinal-regression-v1` | `8ab158b7016c4b30b81ae872e00a40f6` | Evidence only |
| TF-IDF ordinal logistic | `archive/ordinal-logistic/2026-07-19` | `7503f9e` | `ordinal-logistic-v1` | `9db0c255fb9f4c46b7fb7654953d934c` | `sentiment-tfidf-ordinal-logistic` |
| Jina ordinal logistic | `archive/jina-ordinal-logistic/2026-07-19` | `f52ad65` | `jina-ordinal-logistic-v1` | `c030142c6d3e46f78beb730c680a54e8` | `sentiment-jina-v3-ordinal-logistic` |
| Alternate Jina-from-ordinal lineage | `archive/jina-ordinal-from-ordinal/2026-07-19` | `d1e14fb` | Covered by the Jina and ordinal evidence runs | N/A | Evidence only |

The Jina ordinal evidence run records commit `f7f3572`, which contains the completed experiment.
The archive tag points to `f52ad65` so it also preserves the later registry and repository-governance
work. The alternate lineage is retained as a tag for ancestry recovery but does not represent a
separate trained model.

## Registry boundary

- `sentiment-production@champion` is the only model approved for the public service.
- Challenger aliases describe frozen candidates; they do not authorize deployment.
- Benchmark and research records remain reproducible evidence and must not be loaded by the service.
- The local `mlflow.db` and `mlruns/` are intentionally outside Git and require independent backup.
- A Git branch may be deleted only after its archive tag is on `origin` and its evidence row above is
  verified against MLflow.

