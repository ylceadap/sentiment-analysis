# Negative-Class Imbalance Experiment

## Technical summary

**Recommendation:** Keep `submission-v1`; the candidate did not pass every predeclared promotion gate.

The official baseline remains balanced Logistic Regression with argmax. Candidate weights, fold-local random oversampling, and Negative thresholds were selected only from out-of-fold training predictions. The frozen held-out set was never used to select a weight, sampling ratio, or threshold.

## OOF precision-constrained selection

Eligibility required Negative precision ≥ 0.60. Candidates within 0.01 recall of the best eligible recall were resolved by macro-F1 mean, macro-F1 standard deviation, and simplicity.

| name                                | strategy            | class_weight                                  |   negative_multiplier | negative_threshold   |   oof_negative_precision |   oof_negative_recall |   oof_negative_f1_score |   cv_macro_f1_mean |   cv_macro_f1_std |   oof_accuracy | oof_confusion_matrix                              |
|:------------------------------------|:--------------------|:----------------------------------------------|----------------------:|:---------------------|-------------------------:|----------------------:|------------------------:|-------------------:|------------------:|---------------:|:--------------------------------------------------|
| negative_weight_10__0.30            | class_weight        | {"Average": 1, "Negative": 10, "Positive": 1} |                     1 | 0.30                 |                   0.608  |                0.6333 |                  0.6204 |             0.6522 |            0.0145 |         0.6655 | [[1183, 571, 45], [527, 1219, 53], [30, 58, 152]] |
| balanced__argmax                    | class_weight        | balanced                                      |                     1 | argmax               |                   0.7333 |                0.5042 |                  0.5975 |             0.6472 |            0.0269 |         0.6699 | [[1211, 569, 19], [535, 1239, 25], [45, 74, 121]] |
| negative_weight_3__0.20             | class_weight        | {"Average": 1, "Negative": 3, "Positive": 1}  |                     1 | 0.20                 |                   0.6016 |                0.6292 |                  0.6151 |             0.6505 |            0.0108 |         0.6652 | [[1180, 572, 47], [524, 1222, 53], [30, 59, 151]] |
| balanced__0.30                      | class_weight        | balanced                                      |                     1 | 0.30                 |                   0.6203 |                0.6125 |                  0.6164 |             0.6526 |            0.0141 |         0.6678 | [[1193, 565, 41], [527, 1223, 49], [35, 58, 147]] |
| negative_weight_5__0.25             | class_weight        | {"Average": 1, "Negative": 5, "Positive": 1}  |                     1 | 0.25                 |                   0.6125 |                0.6125 |                  0.6125 |             0.6493 |            0.0145 |         0.6647 | [[1182, 575, 42], [526, 1222, 51], [33, 60, 147]] |
| negative_weight_7__0.30             | class_weight        | {"Average": 1, "Negative": 7, "Positive": 1}  |                     1 | 0.30                 |                   0.653  |                0.5958 |                  0.6231 |             0.6535 |            0.0157 |         0.6668 | [[1191, 575, 33], [531, 1225, 43], [35, 62, 143]] |
| random_oversample_negative_x3__0.20 | random_oversampling | none                                          |                     3 | 0.20                 |                   0.6164 |                0.5958 |                  0.6059 |             0.647  |            0.0174 |         0.6647 | [[1187, 572, 40], [529, 1221, 49], [33, 64, 143]] |
| negative_weight_10__0.35            | class_weight        | {"Average": 1, "Negative": 10, "Positive": 1} |                     1 | 0.35                 |                   0.6749 |                0.5708 |                  0.6185 |             0.6518 |            0.0163 |         0.6665 | [[1194, 578, 27], [533, 1227, 39], [37, 66, 137]] |
| balanced__0.35                      | class_weight        | balanced                                      |                     1 | 0.35                 |                   0.7151 |                0.5333 |                  0.611  |             0.652  |            0.0184 |         0.6701 | [[1209, 569, 21], [534, 1235, 30], [43, 69, 128]] |

Full metrics for all 42 combinations are in `artifacts/imbalance_oof_results.csv`. Each row includes per-class precision/recall/F1, balanced accuracy, macro-F1 mean/std, and an OOF confusion matrix.

A chart is intentionally omitted: the 42-row experiment is an audit table with exact precision/recall constraints, and a reduced visual would hide threshold and fold-dispersion details needed for the decision.

## Frozen held-out comparison

| metric             |   baseline |   selected |   delta |
|:-------------------|-----------:|-----------:|--------:|
| accuracy           |     0.6531 |     0.6479 | -0.0052 |
| balanced_accuracy  |     0.6137 |     0.6485 |  0.0348 |
| macro_f1           |     0.6379 |     0.6396 |  0.0017 |
| negative_precision |     0.7209 |     0.5909 | -0.13   |
| negative_recall    |     0.5167 |     0.65   |  0.1333 |
| negative_f1_score  |     0.6019 |     0.619  |  0.0171 |

Per-class held-out metrics:

| label    | metric    |   baseline |   selected |   delta |
|:---------|:----------|-----------:|-----------:|--------:|
| Positive | precision |     0.6682 |     0.6706 |  0.0024 |
| Positive | recall    |     0.6356 |     0.6244 | -0.0111 |
| Positive | f1-score  |     0.6515 |     0.6467 | -0.0048 |
| Average  | precision |     0.6339 |     0.6358 |  0.0018 |
| Average  | recall    |     0.6889 |     0.6711 | -0.0178 |
| Average  | f1-score  |     0.6603 |     0.653  | -0.0073 |
| Negative | precision |     0.7209 |     0.5909 | -0.13   |
| Negative | recall    |     0.5167 |     0.65   |  0.1333 |
| Negative | f1-score  |     0.6019 |     0.619  |  0.0171 |

Baseline confusion matrix: `[[286, 160, 4], [132, 310, 8], [10, 19, 31]]`

Selected confusion matrix: `[[281, 160, 9], [130, 302, 18], [8, 13, 39]]`

Dutch Negative baseline vs selected: precision 0.7209 → 0.5909; recall 0.5345 → 0.6724; F1 0.6139 → 0.6290.

Promotion checks:

```json
{
  "checks": {
    "negative_precision_floor": false,
    "negative_recall_improved": true,
    "macro_f1_guardrail": true,
    "accuracy_guardrail": true
  },
  "promote": false
}
```

The official artifact reports Dutch Negative recall as 31/58 = 0.5345, not 29/58 = 0.5000. Overall Negative recall is 31/60 = 0.5167. These frozen artifact values control the comparison.

## Scope, data, and metric definitions

- Unified training population: 3838 Dutch and English rows; no language-specific model.
- Training labels: `{"Average": 1799, "Negative": 240, "Positive": 1799}`.
- Dutch training slice: Positive 1,678; Average 1,540; Negative 232.
- English training slice: Positive 121; Average 259; Negative 8.
- Frozen training hash: `aa986ef1d8f35ebf232a015fd0e61d3affb1efbe4a589ec4f8653ae01e8ab7c9`.
- Frozen held-out hash: `b76afd73eaeed79bf61903ab8475a08c4565cf523a55e8ae34448d2330e00cbb`.

## Experimental design and leakage controls

- Five fixed language×label-stratified folds were shared by all candidates.
- Random oversampling occurred after TF-IDF transformation inside each fold's training pipeline only.
- No SMOTE and no majority-class undersampling were used.
- OOF probabilities were generated once per base strategy and reused for threshold comparisons.
- Balanced argmax reproduced frozen CV macro-F1: 0.647243512180.
- Held-out lock: `artifacts/imbalance_heldout_lock.json`; it prevents accidental repeat evaluation.

## Limitations and robustness

- Only 240 Negative examples exist in unified training, including just 8 English Negative examples.
- Comparing 42 OOF decision combinations creates selection uncertainty; fold dispersion is reported, and the held-out set is not reused for tuning.
- Weighting, oversampling, and thresholds change decision trade-offs but cannot create missing linguistic coverage.
- Thresholded probabilities are decision scores, not evidence of improved calibration.

## Recommended next steps

1. Keep the frozen submission unless every configured promotion gate passes.
2. For future versions, collect and manually review more real Dutch Negative reviews, especially Negative/Average boundaries, restrained criticism, sarcasm, negation, and mixed praise/criticism.
3. Treat the two held-out English Negative rows as descriptive only; they cannot support a reliable English minority-class conclusion.

## Further questions

- Does the selected policy remain stable after acquiring a materially larger Negative validation sample?
- Which error types account for the remaining Negative→Average mistakes?
