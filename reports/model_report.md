# Model Report

## 1. Executive summary

The final system uses `combined_balanced_ratings`. Selection is based on stratified cross-validation macro-F1 over the training partition, with Negative-class behavior, latency, size, probability support, and explanation feasibility treated as engineering constraints. The held-out test set is used only after selection.

## 2. Dataset description

- Raw rows: 4800
- Raw SHA-256: `2788b987e2c9fa4fd6459a1798a6e7d1dd63ddb5618c10907712d39042cc4be2`
- Original labels: {"Positive": {"count": 2250, "percentage": 46.88}, "Average": {"count": 2250, "percentage": 46.88}, "Negative": {"count": 300, "percentage": 6.25}}
- Text is label-ordered, so sequential splitting is invalid.

## 3. Language composition and unified training policy

- Status counts: {"dutch": 4315, "non_dutch": 485}
- Status by label: {"Positive": {"dutch": 2099, "non_dutch": 151}, "Average": {"dutch": 1926, "non_dutch": 324}, "Negative": {"dutch": 290, "non_dutch": 10}}
- Every deduplicated Dutch and English row is retained in one shared model; no language-specific model is trained.
- Holdout and CV splits jointly stratify detected language and label.
- Detector output is not a gold annotation; manual bounded examples are in `reports/data_audit.md`.

## 4. Data-quality findings

- Exact duplicate extras: 2
- Normalized conflicting groups: 0
- HTML breaks: 2908; zero-width characters: 1261; mojibake candidates: 122.

## 5. Leakage risks

The pipeline prevents normalized duplicate overlap, fits all vectorizers inside CV folds, never uses labels during language detection, and compares explicit ratings retained versus replaced by a neutral token. Ratings remain legitimate review content but may directly encode how source labels were assigned.

## 6. Split methodology

{
  "random_seed": 42,
  "test_size": 0.2,
  "raw_rows": 4800,
  "annotated_rows": 4800,
  "stratification_columns": [
    "detected_language",
    "Label"
  ],
  "training_rows": 3838,
  "test_rows": 960,
  "duplicate_rows_removed": 2,
  "conflicting_groups_removed": 0,
  "train_label_counts": {
    "Positive": 1799,
    "Average": 1799,
    "Negative": 240
  },
  "test_label_counts": {
    "Average": 450,
    "Positive": 450,
    "Negative": 60
  },
  "train_language_counts": {
    "dutch": 3450,
    "english": 388
  },
  "test_language_counts": {
    "dutch": 863,
    "english": 97
  },
  "train_language_label_counts": {
    "dutch::Average": 1540,
    "dutch::Negative": 232,
    "dutch::Positive": 1678,
    "english::Average": 259,
    "english::Negative": 8,
    "english::Positive": 121
  },
  "test_language_label_counts": {
    "dutch::Average": 385,
    "dutch::Negative": 58,
    "dutch::Positive": 420,
    "english::Average": 65,
    "english::Negative": 2,
    "english::Positive": 30
  },
  "train_normalized_sha256": "aa986ef1d8f35ebf232a015fd0e61d3affb1efbe4a589ec4f8653ae01e8ab7c9",
  "test_normalized_sha256": "b76afd73eaeed79bf61903ab8475a08c4565cf523a55e8ae34448d2330e00cbb"
}

## 7–8. Experiment and cross-validation results

| name                             | feature_kind   | class_weight   | mask_ratings   |   cv_accuracy_mean |   cv_accuracy_std |   cv_balanced_accuracy_mean |   cv_balanced_accuracy_std |   cv_macro_precision_mean |   cv_macro_precision_std |   cv_macro_recall_mean |   cv_macro_recall_std |   cv_macro_f1_mean |   cv_macro_f1_std |   cv_weighted_f1_mean |   cv_weighted_f1_std | mlflow_run_id                    |   artifact_size_bytes |
|:---------------------------------|:---------------|:---------------|:---------------|-------------------:|------------------:|----------------------------:|---------------------------:|--------------------------:|-------------------------:|-----------------------:|----------------------:|-------------------:|------------------:|----------------------:|---------------------:|:---------------------------------|----------------------:|
| combined_balanced_ratings        | combined       | balanced       | False          |             0.6699 |            0.0104 |                      0.622  |                     0.0292 |                    0.6895 |                   0.02   |                 0.622  |                0.0292 |             0.6472 |            0.0269 |                0.6689 |               0.0111 | f1a9b3c8edfd44a6984296ac89e7dfff |               2935533 |
| combined_balanced_masked_ratings | combined       | balanced       | True           |             0.6681 |            0.011  |                      0.6207 |                     0.0294 |                    0.6888 |                   0.0171 |                 0.6207 |                0.0294 |             0.6459 |            0.0257 |                0.667  |               0.0116 | b763c3809438467880f75a100b2bbf89 |               2935751 |
| combined_logreg                  | combined       | none           | False          |             0.648  |            0.0089 |                      0.4837 |                     0.0088 |                    0.7646 |                   0.0056 |                 0.4837 |                0.0088 |             0.4922 |            0.014  |                0.6329 |               0.0087 | 94d9c63472da4133ad1b5df3a51dfa77 |               2935580 |
| char_logreg                      | char           | none           | False          |             0.6456 |            0.0058 |                      0.4724 |                     0.005  |                    0.7633 |                   0.0036 |                 0.4724 |                0.005  |             0.4721 |            0.0066 |                0.6283 |               0.0059 | 6163b3e0ac644c488b473e39ccee8329 |               1551708 |
| word_logreg                      | word           | none           | False          |             0.6464 |            0.0127 |                      0.4609 |                     0.0095 |                    0.4978 |                   0.1343 |                 0.4609 |                0.0095 |             0.4474 |            0.0105 |                0.6258 |               0.0124 | d198412f2ef04d0fa958028a69dda657 |               1384649 |
| dummy_prior                      | word           | none           | False          |             0.4685 |            0.0003 |                      0.3333 |                     0      |                    0.1562 |                   0.0001 |                 0.3333 |                0      |             0.2127 |            0.0001 |                0.2989 |               0.0004 | 30653994b91847c2b9445d40ee8aa3b9 |                  2533 |

## 9. Final held-out test metrics

- Accuracy: 0.6531
- Balanced accuracy: 0.6137
- Macro precision: 0.6744
- Macro recall: 0.6137
- Macro-F1: 0.6379
- Weighted F1: 0.6525
- Log loss: 0.7216
- Multiclass Brier score: 0.4462
- Expected calibration error (10 bins): 0.0470
- Mean prediction confidence: 0.6103

These probability metrics are descriptive evidence on the held-out set. Logistic Regression supplies native probabilities, but no separate calibration model was fitted; the calibration estimate is not an operational guarantee.

### Held-out metrics by detected language

| detected_language   |   support |   accuracy |   balanced_accuracy |   macro_f1 |   negative_f1 |   negative_support |   log_loss |
|:--------------------|----------:|-----------:|--------------------:|-----------:|--------------:|-------------------:|-----------:|
| dutch               |       863 |     0.6489 |              0.6158 |     0.6381 |        0.6139 |                 58 |     0.729  |
| english             |        97 |     0.6907 |              0.3615 |     0.3289 |        0      |                  2 |     0.6556 |

Language slices evaluate the same shared model and are not separately trained models. English results require extra caution because the source has only 485 English rows and 10 English Negative rows; the held-out English-Negative support is especially small.

## 10. Per-class metrics

| label    |   precision |   recall |     f1 |   support |
|:---------|------------:|---------:|-------:|----------:|
| Positive |      0.6682 |   0.6356 | 0.6515 |       450 |
| Average  |      0.6339 |   0.6889 | 0.6603 |       450 |
| Negative |      0.7209 |   0.5167 | 0.6019 |        60 |

## 11. Confusion matrix

Rows are true labels and columns are predicted labels.

|          |   Positive |   Average |   Negative |
|:---------|-----------:|----------:|-----------:|
| Positive |        286 |       160 |          4 |
| Average  |        132 |       310 |          8 |
| Negative |         10 |        19 |         31 |

## 12. Negative-class analysis

Negative recall is 0.5167 and Negative F1 is 0.6019. This class has the smallest support, so point estimates should be interpreted with more uncertainty than Positive or Average results.

## 13. Rating-leakage experiment

| name                             | feature_kind   | class_weight   | mask_ratings   |   cv_accuracy_mean |   cv_accuracy_std |   cv_balanced_accuracy_mean |   cv_balanced_accuracy_std |   cv_macro_precision_mean |   cv_macro_precision_std |   cv_macro_recall_mean |   cv_macro_recall_std |   cv_macro_f1_mean |   cv_macro_f1_std |   cv_weighted_f1_mean |   cv_weighted_f1_std | mlflow_run_id                    |   artifact_size_bytes |
|:---------------------------------|:---------------|:---------------|:---------------|-------------------:|------------------:|----------------------------:|---------------------------:|--------------------------:|-------------------------:|-----------------------:|----------------------:|-------------------:|------------------:|----------------------:|---------------------:|:---------------------------------|----------------------:|
| combined_balanced_ratings        | combined       | balanced       | False          |             0.6699 |            0.0104 |                      0.622  |                     0.0292 |                    0.6895 |                   0.02   |                 0.622  |                0.0292 |             0.6472 |            0.0269 |                0.6689 |               0.0111 | f1a9b3c8edfd44a6984296ac89e7dfff |               2935533 |
| combined_balanced_masked_ratings | combined       | balanced       | True           |             0.6681 |            0.011  |                      0.6207 |                     0.0294 |                    0.6888 |                   0.0171 |                 0.6207 |                0.0294 |             0.6459 |            0.0257 |                0.667  |               0.0116 | b763c3809438467880f75a100b2bbf89 |               2935751 |

The paired rows differ only in rating masking. The final choice follows CV macro-F1 and Negative-class evidence rather than automatically retaining the higher-leakage signal.

## 14. Error analysis

Excerpts are deliberately capped and whitespace-normalized.

| actual   | predicted   | detected_language   | excerpt                                                                                                                                                          |
|:---------|:------------|:--------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Average  | Positive    | dutch               | Het is een vreemd, maar op de een of andere manier indrukwekkend verhaal over liefde. Persoonlijk kom ik in het echte leven nog nooit zo'n twist-off-verhaal teg |
| Positive | Average     | dutch               | Met enige nieuwsgierigheid heb ik deze film bekeken. Ik wilde zien of 1) Paul Muni Chinees kon spelen en 2) Luise Rainer haar Oscar verdiende. Ik verliet de fil |
| Average  | Positive    | dutch               | Heb deze film net op dvd gezien en vond het acteerwerk erg goed, maar het algehele verhaal liet veel te wensen over. Deze film heeft dezelfde textuur en uitstra |
| Positive | Average     | dutch               | Henry Fool van Hal Hartley was een onafhankelijk filmmeesterwerk en zeker zijn beste werk. Het heeft een enorme karakterdiepte, subtiele, gecompliceerde dialoge |
| Average  | Positive    | dutch               | Ik heb deze film onlangs gezien in mijn International Business-klas. Ik had niets anders verwacht dan weer een saaie documentaire (om niet te zeggen dat ik niet |
| Positive | Average     | dutch               | Dit is een betere bewerking van het boek dan die met Paltrow (hoewel ik die ook leuk vond). Het is niet zozeer dat Beckinsale beter is – ze zijn allebei erg goe |
| Positive | Average     | dutch               | Ik vond deze film erg vermakelijk. Ik heb deze film met mijn vrouw bekeken VOORDAT ik mijn eerste kind kreeg. Daarom zag ik het niet als gewoon familie-entertai |
| Positive | Average     | english             | One shot, one kill, no exceptions. A must see if you are into marines or snipers. two big thumbs up! Great overall storyline, great camera work, good drama, act |
| Average  | Positive    | dutch               | De mannen kunnen genieten van Lollo, als ze dat willen (of haar lollos – ze heeft haar naam gegeven aan jargontermen voor borsten in het Frans), maar de dames h |
| Average  | Positive    | dutch               | Als Jack Nicholsons regiedebuut Drive, laat He Said op zijn minst zien dat hij een begaafd acteursregisseur is. Zelfs als het verhaal de weg lijkt te verliezen  |
| Positive | Average     | dutch               | De Duitse middeleeuwse saga van Fritz Lang gaat verder in Die Nibelungen: Kriemhilds Rache (1924). Kriemhild (Margarete Schoen) wil haar vermoorde echtgenoot Si |
| Average  | Positive    | dutch               | Er zijn weinig films of films die ik door de jaren heen als favorieten beschouw. De Evangelieweg was er één van. Ik heb dit als jonge tiener gezien en zou het g |

Observed failures may combine ambiguous sentiment, label noise, sparse Negative evidence, cross-language imbalance, and bag-of-ngrams limitations. Feature contributions do not establish causal reasons.

## 15–16. Inference latency, size, and load time

| path                                |   iterations |   mean_ms |   p50_ms |   p95_ms |   max_ms |
|:------------------------------------|-------------:|----------:|---------:|---------:|---------:|
| normalization_plus_model_prediction |          100 |     3.906 |    3.819 |    6.288 |    6.569 |
| language_detection_plus_model       |          100 |     5.143 |    4.835 |    8.174 |    8.945 |
| service_end_to_end                  |          100 |     5.666 |    5.32  |    8.807 |    9.711 |
| explanation_enabled                 |          100 |     6.499 |    5.95  |    9.477 |   10.684 |
| http_classify                       |           50 |     8.045 |    7.571 |   11.146 |   11.778 |

- Model artifact bytes: 2935649
- Model SHA-256: `32ec6bc66d70c26f50bc7b6f495d0852cdd1ee0fd68cbff97d823b34370bf836`
- Language-detector constructor (ms): 0.368613051250577
- Cold first language inference/model load (ms): 1610.0913219852373
- Cold load time (ms): 630.5980730103329

## 17. Prediction explanation

The API can return active word n-grams supporting/opposing the selected class and a separate technical list of character n-grams. Contributions are local linear associations, not causal or semantic explanations.

## 18. Final model-selection reasoning

Selected specification: `{"name": "combined_balanced_ratings", "feature_kind": "combined", "class_weight": "balanced", "mask_ratings": false, "dummy": false}`. It is a CPU-friendly sparse linear model with native multiclass probabilities, fast batch-size-one inference, a single serialized normalization/feature/classifier pipeline, and inspectable coefficients.

## 19. Limitations

- Language detection is uncertain for short, mixed, or named-entity-heavy text.
- English predictions are supported but less reliable because English supervision is limited and highly class-imbalanced.
- Sparse n-grams do not deeply model negation scope, irony, or long-range composition.
- Negative has limited raw and held-out support.
- The supplied labels and their construction were not independently verified.
- No temporal/source fields exist for drift or lineage analysis.

## 20. Sensible production improvements

Collect adjudicated Dutch and English labels, monitor metrics and drift by language/label, calibrate and threshold behavior against business costs, add safe model registry promotion, and compare a compact multilingual transformer only after establishing a larger trustworthy benchmark.
