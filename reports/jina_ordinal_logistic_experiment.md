# Jina v3 ordinal-logistic experiment

## Decision

Selected OOF candidate: `jina_v3_classification__ordinal_composed_argmax_C_2`.

CV Macro-F1 is 0.7299, versus 0.6472 for the official TF-IDF baseline. Negative precision/recall/F1 are 0.8089/0.7583/0.7828.

Production model and held-out test were not changed. This is a Colab/GPU research experiment using frozen Jina embeddings plus ordinal boundary classifiers.

## Promotion gates

- PASS - `macro_f1`
- PASS - `negative_precision`
- PASS - `negative_recall`
- PASS - `accuracy`
- PASS - `stability`

Metric gates passed: `True`.

## Method

- Recreated and hash-verified the frozen 3,838-row training split and 960-row reserved holdout split.
- Encoded Dutch and English training reviews together with revision-pinned Jina v3 classification embeddings.
- Compared balanced multiclass Logistic Regression against two calibrated ordinal boundaries: `Negative < Average < Positive`.
- Selected all C values and boundary thresholds only from training-set out-of-fold predictions.
- Did not evaluate the reserved holdout rows, replace `artifacts/model.joblib`, or train a separate English model.

## Top OOF configurations

| name                                                     | model_type                                   |   cv_macro_f1_mean |   cv_macro_f1_std |   oof_accuracy |   negative_precision |   negative_recall |   negative_f1 |   ordinal_mae |   quadratic_weighted_kappa |   severe_error_rate |
|:---------------------------------------------------------|:---------------------------------------------|-------------------:|------------------:|---------------:|---------------------:|------------------:|--------------:|--------------:|---------------------------:|--------------------:|
| jina_v3_classification__ordinal_composed_argmax_C_2      | jina_embedding_two_boundary_ordinal_logistic |             0.7299 |            0.0083 |         0.7084 |               0.8089 |            0.7583 |        0.7828 |        0.2952 |                     0.5817 |              0.0036 |
| jina_v3_classification__ordinal_crossfit_threshold_C_2   | jina_embedding_two_boundary_ordinal_logistic |             0.7289 |            0.0122 |         0.7071 |               0.8349 |            0.7375 |        0.7832 |        0.2952 |                     0.5813 |              0.0023 |
| jina_v3_classification__ordinal_crossfit_threshold_C_1   | jina_embedding_two_boundary_ordinal_logistic |             0.7277 |            0.0151 |         0.7045 |               0.8219 |            0.75   |        0.7843 |        0.2978 |                     0.5798 |              0.0023 |
| jina_v3_classification__ordinal_composed_argmax_C_1      | jina_embedding_two_boundary_ordinal_logistic |             0.725  |            0.0145 |         0.704  |               0.8009 |            0.7542 |        0.7768 |        0.2994 |                     0.5771 |              0.0034 |
| jina_v3_classification__ordinal_composed_argmax_C_0.5    | jina_embedding_two_boundary_ordinal_logistic |             0.7215 |            0.0162 |         0.6985 |               0.8044 |            0.7542 |        0.7785 |        0.3043 |                     0.5717 |              0.0029 |
| jina_v3_classification__ordinal_crossfit_threshold_C_0.5 | jina_embedding_two_boundary_ordinal_logistic |             0.7183 |            0.0192 |         0.6944 |               0.8165 |            0.7417 |        0.7773 |        0.308  |                     0.5646 |              0.0023 |
| jina_v3_classification__multiclass_balanced_C_2          | jina_embedding_multiclass_logistic           |             0.7072 |            0.0115 |         0.7011 |               0.6006 |            0.9083 |        0.7231 |        0.3098 |                     0.579  |              0.0109 |
| jina_v3_classification__multiclass_balanced_C_1          | jina_embedding_multiclass_logistic           |             0.7071 |            0.0114 |         0.7035 |               0.5882 |            0.9167 |        0.7166 |        0.308  |                     0.5826 |              0.0115 |
| jina_v3_classification__multiclass_balanced_C_0.5        | jina_embedding_multiclass_logistic           |             0.7014 |            0.0152 |         0.6988 |               0.5736 |            0.925  |        0.7081 |        0.3132 |                     0.5781 |              0.012  |
| official_tfidf_baseline                                  | tfidf_word_char_logreg                       |             0.6472 |            0.0269 |         0.6699 |               0.7333 |            0.5042 |        0.5975 |      nan      |                   nan      |            nan      |

## Limitations

Jina Embeddings v3 is CC-BY-NC-4.0, so this branch is non-commercial research evidence. English slice metrics remain directional because English Negative support is tiny. A final promotion decision still needs a newly collected blind test.
