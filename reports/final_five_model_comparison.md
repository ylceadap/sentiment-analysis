# Final five-model comparison

## Interpretation

All five frozen candidates were evaluated against the same 960-row
held-out split. The split was never used to fit any of the five models, but it has appeared in prior
project reports. Therefore this is a **reused-heldout presentation comparison**, not a new blind test
and not authorization to tune parameters after seeing the ranking.

Ranking metric: held-out Macro-F1, descending.

|   rank | model                     |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |   negative_precision |   negative_recall |   negative_f1 |   ordinal_mae |   quadratic_weighted_kappa |   severe_error_rate | log_loss   | multiclass_brier_score   | expected_calibration_error_10_bin   |
|-------:|:--------------------------|-----------:|--------------------:|-----------:|--------------:|---------------------:|------------------:|--------------:|--------------:|---------------------------:|--------------------:|:-----------|:-------------------------|:------------------------------------|
|      1 | DeepSeek V4 Flash 24-shot |     0.7208 |              0.7774 |     0.7506 |        0.7144 |               0.7746 |            0.9167 |        0.8397 |        0.2833 |                     0.6238 |              0.0042 | —          | —                        | —                                   |
|      2 | Jina Ordinal              |     0.6896 |              0.7215 |     0.7104 |        0.6894 |               0.7273 |            0.8    |        0.7619 |        0.3156 |                     0.5604 |              0.0052 | 0.6533     | 0.4051                   | 0.0172                              |
|      3 | Jina Logistic             |     0.6719 |              0.733  |     0.6715 |        0.6718 |               0.5408 |            0.8833 |        0.6709 |        0.3385 |                     0.5517 |              0.0104 | 0.6651     | 0.429                    | 0.0368                              |
|      4 | TF-IDF Ordinal            |     0.65   |              0.6404 |     0.6406 |        0.6461 |               0.6379 |            0.6167 |        0.6271 |        0.3625 |                     0.4517 |              0.0125 | 0.6889     | 0.4414                   | 0.0342                              |
|      5 | Current Production TF-IDF |     0.6531 |              0.6137 |     0.6379 |        0.6525 |               0.7209 |            0.5167 |        0.6019 |        0.3615 |                     0.4388 |              0.0146 | 0.7216     | 0.4462                   | 0.047                               |

## Governance

- Selected for presentation: exactly the five rows above.
- Production remains `Current Production TF-IDF`; this comparison does not change the champion.
- Jina models are research-only under CC-BY-NC-4.0 and use a pinned external encoder.
- DeepSeek is an archived external API result; provider weights are not stored locally.
- All other registered models remain benchmark, ablation, baseline, or test-only evidence.
- `parameters_frozen_before_evaluation=true`; `post_evaluation_tuning_allowed=false`.
