# RobBERT v2 Experiments

Both experiment stages used the pinned `pdelobelle/robbert-v2-dutch-base` revision
`271b8bf12b7e429434ce953efb432e8373e84453` on Colab Tesla T4. Dutch and English reviews remained
in one model. Candidate and epoch selection used training-only validation; the 960-row test split
was evaluated only after each candidate was frozen.

## Initial paired fine-tuning

| Candidate | Selected epoch | Macro-F1 | Accuracy | Negative precision | Negative recall | Ordinal MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RobBERT Logistic | 2 | 0.6252 | 0.6354 | 0.5873 | 0.6167 | 0.3792 |
| RobBERT CORAL ordinal | 3 | 0.3460 | 0.4771 | 0.2347 | 0.7667 | 0.5771 |

Logistic is retained as one of the seven presentation models. CORAL collapsed the Average class,
predicting no Average examples, and remains test-only. The verified evidence is stored in
`artifacts/robbert_v2/` and MLflow run `74b1a804f3694d66a0a8c7c9360c1e6d`. The approximately
934 MB of candidate weights are not duplicated locally.

## Mixed-language improvement sweep

Seven token/loss candidates were screened on training fold 0. Two candidates satisfying Average
recall of at least 0.40 were confirmed with 5-fold × 3-seed validation. The frozen winner was refit
on all 3,838 training rows as a three-seed ensemble.

| Stage | Candidate | Macro-F1 | Accuracy | Negative precision | Negative recall |
| --- | --- | ---: | ---: | ---: | ---: |
| OOF, 5-fold × 3-seed | Head-tail 512 + mild class weights | 0.6938 | 0.6954 | 0.7026 | 0.6792 |
| OOF, 5-fold × 3-seed | Head-tail 512 + cross-entropy | 0.6879 | 0.6957 | 0.7051 | 0.6375 |
| Test, 3-seed ensemble | **Head-tail 512 + mild class weights** | **0.6615** | **0.6771** | **0.6939** | **0.5667** |

The ensemble improves on RobBERT Logistic by 0.0363 Macro-F1 and on Production TF-IDF by 0.0236,
but remains below Jina Logistic, Jina Ordinal, and the external LLM result. It is included in the
seven-model presentation as a challenger evaluation; Production and the UI remain unchanged.

The verified bundle contains predictions, trial histories, configuration, checksums, and three
approximately 467 MB seed weights in
`MyDrive/Workspace365_assignment/robbert_improvement`. Compact evidence is retained in
`artifacts/robbert_improvement/` and MLflow run `62a5a68d9e1d48c6832b94641524f53e`.

## Interpretation limits

The supplied test partition has appeared in earlier project reports, so these results are comparative
evidence rather than a new blind promotion test. Negative has only 60 test examples, and English
Negative has support two; minority-language conclusions therefore remain uncertain.
