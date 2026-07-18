# Jina embedding experiment validation

## Overall assessment: Share with caveats

The Colab experiment is methodologically sound enough to justify a separate research fine-tuning experiment. It is not evidence for replacing the official TF-IDF model or deploying Jina v3 in production.

## Verified checks

- The CSV contains one official baseline plus 30 unique Jina candidates from the declared 3 × 2 × 5 grid.
- The frozen split hashes match the configuration: 3,838 training rows and 960 untouched held-out rows.
- `heldout_evaluated` and `official_model_replaced` are both `false`.
- Recomputing the five gates from the CSV reproduces five passes.
- The selected candidate is unique and uses Jina v3 revision `ab036b023d30b4d1138c4c3bfa9f0c445ab455d6`.
- The observed remote-code dependency is recorded at `845308d0fd72a8406a3e378450e1a09522790419`.

## Calculation spot-checks

Compared with the official OOF baseline, the selected candidate changes:

- CV Macro-F1: 0.6472 → 0.7108 (+0.0636)
- OOF accuracy: 0.6699 → 0.7024 (+0.0326)
- Negative precision: 0.7333 → 0.6196 (-0.1137)
- Negative recall: 0.5042 → 0.8958 (+0.3917)
- Negative F1: 0.5975 → 0.7325 (+0.1350)
- CV Macro-F1 standard deviation: 0.0269 → 0.0122 (-0.0147)

## Required caveats

- These are training-set out-of-fold results, not final generalization evidence.
- No current held-out metrics were computed; a newly collected blind test is still required after model selection.
- The English slice has only 388 rows and 8 English Negative rows, so its 0.6615 Macro-F1 is directional only.
- Jina Embeddings v3 is CC-BY-NC-4.0. Metric gates passing does not make this configuration eligible for commercial production deployment.
- `trust_remote_code` executes third-party model code. Exact observed revisions are recorded for provenance and must be reviewed again before any later run.
