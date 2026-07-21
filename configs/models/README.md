# Model configurations

Each file defines one model family independently of Git branches. Research configurations may write
evidence to MLflow, but they do not change the production champion without a separate promotion
decision.

| Configuration | Model family | Lifecycle |
| --- | --- | --- |
| `tfidf_logreg.yaml` | Word/character TF-IDF plus multiclass logistic regression | Production champion |
| `tfidf_ordinal.yaml` | TF-IDF plus ordinal logistic boundaries | Frozen challenger |
| `jina_logreg.yaml` | Jina v3 embeddings plus multiclass logistic regression | Research only |
| `jina_ordinal.yaml` | Jina v3 embeddings plus ordinal logistic boundaries | Research only |
| `llm_advisor.yaml` | External DeepSeek frozen 24-shot runtime profile | Advisory only; same prompt as the held-out evidence |
| `robbert_v2.yaml` | Paired end-to-end RobBERT multiclass-logistic and CORAL-ordinal heads | Logistic: presentation evidence; CORAL: test-only; Colab GPU |
| `robbert_improvement.yaml` | Mixed-language RobBERT input/loss sweep with repeated CV | Challenger evaluation; Colab GPU |

Artifact lifecycle is additionally recorded in MLflow as `deployable`, `reproducible`, or
`evidence-only`. These terms describe what is physically preserved; deployment eligibility remains a
separate governance decision.
