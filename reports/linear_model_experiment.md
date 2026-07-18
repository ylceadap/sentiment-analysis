# Linear Model Experiment

## Scope and leakage control

- Branch-only experiment; `main` and `submission-v1` remain unchanged.
- Git branch: `experiment/linear-models`; commit: `0ba52f8960c7d72c2e8b3102009a839ec5aaa809`.
- Working tree dirty when experiment started: **false**.
- Source rows: 4800; training rows used: 3838.
- Frozen training hash: `aa986ef1d8f35ebf232a015fd0e61d3affb1efbe4a589ec4f8653ae01e8ab7c9`.
- Frozen held-out hash: `b76afd73eaeed79bf61903ab8475a08c4565cf523a55e8ae34448d2330e00cbb`.
- Held-out labels evaluated: **no**.
- Candidate selection uses the same language×label-stratified 5-fold indices.
- Frozen Logistic C=1 CV macro-F1: 0.647243512180.
- Reproduced Logistic C=1 CV macro-F1: 0.647243512180.

## Results

| name            | classifier_kind     |   regularization_c |   cv_macro_f1_mean |   cv_macro_f1_std |   oof_negative_f1 |   oof_dutch_macro_f1 |   oof_english_macro_f1 |   cv_fit_seconds |
|:----------------|:--------------------|-------------------:|-------------------:|------------------:|------------------:|---------------------:|-----------------------:|-----------------:|
| logreg_c0.5     | logistic_regression |                0.5 |             0.6536 |            0.024  |            0.6128 |               0.6575 |                 0.2827 |          78.3377 |
| logreg_c1       | logistic_regression |                1   |             0.6472 |            0.0269 |            0.5975 |               0.6524 |                 0.2828 |          83.7142 |
| logreg_c2       | logistic_regression |                2   |             0.6447 |            0.023  |            0.5985 |               0.6504 |                 0.3006 |          91.9008 |
| logreg_c4       | logistic_regression |                4   |             0.6434 |            0.0254 |            0.5974 |               0.6496 |                 0.3104 |         101.907  |
| linear_svc_c0.5 | linear_svc          |                0.5 |             0.6325 |            0.0307 |            0.5668 |               0.6382 |                 0.3181 |          68.1423 |
| linear_svc_c1   | linear_svc          |                1   |             0.6234 |            0.0267 |            0.5475 |               0.6283 |                 0.3398 |          52.3161 |
| linear_svc_c2   | linear_svc          |                2   |             0.6132 |            0.024  |            0.526  |               0.6166 |                 0.3631 |          78.3908 |

## Predeclared decision

```json
{
  "baseline": "logreg_c1",
  "best_candidate": "logreg_c0.5",
  "macro_f1_improvement": 0.006332712563733134,
  "checks": {
    "macro_f1_improvement": false,
    "dutch_macro_f1_guardrail": true,
    "negative_f1_guardrail": true
  },
  "promote": false
}
```

**Recommendation:** Do not replace `submission-v1`; no candidate passed all predeclared gates.

The English slice is descriptive and small. LinearSVC candidates do not provide native probabilities; if one ever passes the CV gates, probability calibration and its latency must be evaluated before API promotion.
