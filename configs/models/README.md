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
| `llm_advisor.yaml` | External DeepSeek recommendation | Advisory only |

The legacy paths `configs/embedding_experiment.yaml` and
`configs/jina_ordinal_logistic.yaml` are compatibility links to these canonical files.
