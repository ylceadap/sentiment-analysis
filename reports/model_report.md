# Model Report

## 1. Executive summary

The final system uses `combined_balanced_ratings`. Selection is based on stratified cross-validation macro-F1 over the training partition, with Negative-class behavior, latency, size, probability support, and explanation feasibility treated as engineering constraints. The held-out test set is used only after selection.

## 2. Dataset description

- Raw rows: 4800
- Raw SHA-256: `2788b987e2c9fa4fd6459a1798a6e7d1dd63ddb5618c10907712d39042cc4be2`
- Original labels: {"Positive": {"count": 2250, "percentage": 46.88}, "Average": {"count": 2250, "percentage": 46.88}, "Negative": {"count": 300, "percentage": 6.25}}
- Text is label-ordered, so sequential splitting is invalid.

## 3. Language-filter results

- Status counts: {"dutch": 4315, "non_dutch": 485}
- Status by label: {"Positive": {"dutch": 2099, "non_dutch": 151}, "Average": {"dutch": 1926, "non_dutch": 324}, "Negative": {"dutch": 290, "non_dutch": 10}}
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
  "confident_dutch_rows": 4315,
  "training_rows": 3450,
  "test_rows": 863,
  "duplicate_rows_removed": 2,
  "conflicting_groups_removed": 0,
  "train_label_counts": {
    "Positive": 1678,
    "Average": 1540,
    "Negative": 232
  },
  "test_label_counts": {
    "Positive": 420,
    "Average": 385,
    "Negative": 58
  },
  "train_normalized_sha256": "e77ab4e1b645624b9d644fb7e447032371088073b6bd0390652e509a9736383f",
  "test_normalized_sha256": "ba03aa1262bb6ce2f150b801944c0789b7bfaa6fe7da595afd634645a998aa1c"
}

## 7–8. Experiment and cross-validation results

| name                             | feature_kind   | class_weight   | mask_ratings   |   cv_accuracy_mean |   cv_accuracy_std |   cv_balanced_accuracy_mean |   cv_balanced_accuracy_std |   cv_macro_precision_mean |   cv_macro_precision_std |   cv_macro_recall_mean |   cv_macro_recall_std |   cv_macro_f1_mean |   cv_macro_f1_std |   cv_weighted_f1_mean |   cv_weighted_f1_std | mlflow_run_id                    |   artifact_size_bytes |
|:---------------------------------|:---------------|:---------------|:---------------|-------------------:|------------------:|----------------------------:|---------------------------:|--------------------------:|-------------------------:|-----------------------:|----------------------:|-------------------:|------------------:|----------------------:|---------------------:|:---------------------------------|----------------------:|
| combined_balanced_ratings        | combined       | balanced       | False          |             0.6759 |            0.0086 |                      0.6218 |                     0.0154 |                    0.7127 |                   0.0229 |                 0.6218 |                0.0154 |             0.6544 |            0.0174 |                0.6752 |               0.0089 | cb9c813dfbb344a4999df9b631a05130 |               2922094 |
| combined_balanced_masked_ratings | combined       | balanced       | True           |             0.6725 |            0.0121 |                      0.6169 |                     0.0164 |                    0.713  |                   0.0208 |                 0.6169 |                0.0164 |             0.6507 |            0.0173 |                0.6717 |               0.0123 | 4033bf3420be49079abe3cc3e84b7fea |               2923005 |
| combined_logreg                  | combined       | none           | False          |             0.6557 |            0.0093 |                      0.4888 |                     0.0107 |                    0.7687 |                   0.0061 |                 0.4888 |                0.0107 |             0.4946 |            0.0157 |                0.6387 |               0.0097 | 973aed7246644df58cf4dd747d95ea3c |               2922191 |
| char_logreg                      | char           | none           | False          |             0.647  |            0.0047 |                      0.4739 |                     0.0075 |                    0.7302 |                   0.0695 |                 0.4739 |                0.0075 |             0.4716 |            0.0129 |                0.628  |               0.0052 | 90306e871b8e46a6a4cffa1c1cfb84a3 |               1537807 |
| word_logreg                      | word           | none           | False          |             0.6394 |            0.0046 |                      0.4572 |                     0.0032 |                    0.4923 |                   0.1316 |                 0.4572 |                0.0032 |             0.4427 |            0.0049 |                0.617  |               0.0039 | 9eb0a62588ac458f8c9ed4a357095a6a |               1384822 |
| dummy_prior                      | word           | none           | False          |             0.4864 |            0.0007 |                      0.3333 |                     0      |                    0.1621 |                   0.0002 |                 0.3333 |                0      |             0.2181 |            0.0002 |                0.3183 |               0.0008 | 0482bdc2c0ca4e1b8aab1ee61fa6a809 |                  2549 |

## 9. Final held-out test metrics

- Accuracy: 0.6477
- Balanced accuracy: 0.6055
- Macro precision: 0.6709
- Macro recall: 0.6055
- Macro-F1: 0.6311
- Weighted F1: 0.6474
- Log loss: 0.7318
- Multiclass Brier score: 0.4532
- Expected calibration error (10 bins): 0.0491
- Mean prediction confidence: 0.6016

These probability metrics are descriptive evidence on the held-out set. Logistic Regression supplies native probabilities, but no separate calibration model was fitted; the calibration estimate is not an operational guarantee.

## 10. Per-class metrics

| label    |   precision |   recall |     f1 |   support |
|:---------|------------:|---------:|-------:|----------:|
| Positive |      0.6731 |   0.6619 | 0.6675 |       420 |
| Average  |      0.6146 |   0.6545 | 0.634  |       385 |
| Negative |      0.725  |   0.5    | 0.5918 |        58 |

## 11. Confusion matrix

Rows are true labels and columns are predicted labels.

|          |   Positive |   Average |   Negative |
|:---------|-----------:|----------:|-----------:|
| Positive |        278 |       138 |          4 |
| Average  |        126 |       252 |          7 |
| Negative |          9 |        20 |         29 |

## 12. Negative-class analysis

Negative recall is 0.5000 and Negative F1 is 0.5918. This class has the smallest support, so point estimates should be interpreted with more uncertainty than Positive or Average results.

## 13. Rating-leakage experiment

| name                             | feature_kind   | class_weight   | mask_ratings   |   cv_accuracy_mean |   cv_accuracy_std |   cv_balanced_accuracy_mean |   cv_balanced_accuracy_std |   cv_macro_precision_mean |   cv_macro_precision_std |   cv_macro_recall_mean |   cv_macro_recall_std |   cv_macro_f1_mean |   cv_macro_f1_std |   cv_weighted_f1_mean |   cv_weighted_f1_std | mlflow_run_id                    |   artifact_size_bytes |
|:---------------------------------|:---------------|:---------------|:---------------|-------------------:|------------------:|----------------------------:|---------------------------:|--------------------------:|-------------------------:|-----------------------:|----------------------:|-------------------:|------------------:|----------------------:|---------------------:|:---------------------------------|----------------------:|
| combined_balanced_ratings        | combined       | balanced       | False          |             0.6759 |            0.0086 |                      0.6218 |                     0.0154 |                    0.7127 |                   0.0229 |                 0.6218 |                0.0154 |             0.6544 |            0.0174 |                0.6752 |               0.0089 | cb9c813dfbb344a4999df9b631a05130 |               2922094 |
| combined_balanced_masked_ratings | combined       | balanced       | True           |             0.6725 |            0.0121 |                      0.6169 |                     0.0164 |                    0.713  |                   0.0208 |                 0.6169 |                0.0164 |             0.6507 |            0.0173 |                0.6717 |               0.0123 | 4033bf3420be49079abe3cc3e84b7fea |               2923005 |

The paired rows differ only in rating masking. The final choice follows CV macro-F1 and Negative-class evidence rather than automatically retaining the higher-leakage signal.

## 14. Error analysis

Excerpts are deliberately capped and whitespace-normalized.

| actual   | predicted   | excerpt                                                                                                                                                          |
|:---------|:------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Positive | Average     | Ik ben helemaal niet kieskeurig als het om horrorfilms gaat, en ik ben bereid ze vrijwel allemaal te bekijken. Dat betekent niet dat ik bereid ben veel ervan op |
| Positive | Average     | vreselijk onderschat met Matt Dillon en Tom Skerritt, goede achtergrond voor een solide verhaal en enkele memorabele regels, goed geacteerd en goed gecast, Tomm |
| Negative | Average     | Wat de originele Killer Tomatoes leuk maakte, was dat het werd gemaakt door mensen zonder budget, die maar een paar dagen gek waren...<br /><br />Dit was iets m |
| Positive | Average     | Als iemand die in de buurt van Buffalo, New York woont, scoorde deze film bij mij al punten voordat ik hem zelfs maar zag, aangezien het verhaal hier is gebasee |
| Negative | Positive    | Het moet het meest oubollige tv-programma op de ether zijn. Dit is waarschijnlijk een ontsnapping voor Jim Belushi en al zijn slechte films. Zijn broer zoog al  |
| Average  | Positive    | Oscarwinnaar Robert Redford (Beste Regisseur. Ordinary People 1980) legt de majesteit van de wildernis van Montana en de kracht van de Amerikaanse familie vast  |
| Average  | Positive    | GÃ³mez Pereira is verantwoordelijk voor enkele van de meest verachtelijke komedies van de nieuwste Spaanse cinema (kijk maar eens naar zijn curriculum vitae), d |
| Positive | Average     | Ik vond dit een uiterst charmante film. Het verhaal lijkt een nauwelijks verhulde autobiografie van John Waters te zijn: Peckers grootste gave is zijn vermogen  |
| Positive | Average     | Op het eerste gezicht lijkt deze film niet bepaald geweldig. Een Bette Davis-film met slechts 166 stemmen op IMDb en een beoordeling van 6,5 moet immers een beh |
| Positive | Average     | De vredestichter afslachten? Blijkbaar. Na het gewelddadige begin waarin Spike, Tom en Jerry allemaal naar elkaar zwaaien, stopt Butch en wil weten waarom. Dat  |
| Average  | Positive    | Heb deze film net op dvd gezien en vond het acteerwerk erg goed, maar het algehele verhaal liet veel te wensen over. Deze film heeft dezelfde textuur en uitstra |
| Positive | Average     | Dit is de meest meeslepende en uitmuntende prestatie die Robert Taylor ooit heeft gegeven. Het overtreft zelfs zijn geweldige optreden als "Johnny Eager", veert |

Observed failures may combine ambiguous sentiment, label noise, sparse Negative evidence, language-detection errors, and bag-of-ngrams limitations. Feature contributions do not establish causal reasons.

## 15–16. Inference latency, size, and load time

| path                                |   iterations |   mean_ms |   p50_ms |   p95_ms |   max_ms |
|:------------------------------------|-------------:|----------:|---------:|---------:|---------:|
| normalization_plus_model_prediction |          100 |     4.869 |    4.509 |    9.132 |   13.393 |
| language_detection_plus_model       |          100 |     6.592 |    6.051 |   10.556 |   16.936 |
| service_end_to_end                  |          100 |     6.219 |    5.752 |    9.716 |   10.891 |
| explanation_enabled                 |          100 |    10.467 |    7.549 |   15.804 |  179.483 |
| http_classify                       |           50 |     8.121 |    7.78  |   11.274 |   11.451 |

- Model artifact bytes: 2922221
- Model SHA-256: `0c193ceb866cd795bc3da6012055079d5448807d7e6c3824571400c1f5af3c65`
- Language-detector constructor (ms): 0.3287718864157796
- Cold first language inference/model load (ms): 1815.875981003046
- Cold load time (ms): 657.5204480905086

## 17. Prediction explanation

The API can return active word n-grams supporting/opposing the selected class and a separate technical list of character n-grams. Contributions are local linear associations, not causal or semantic explanations.

## 18. Final model-selection reasoning

Selected specification: `{"name": "combined_balanced_ratings", "feature_kind": "combined", "class_weight": "balanced", "mask_ratings": false, "dummy": false}`. It is a CPU-friendly sparse linear model with native multiclass probabilities, fast batch-size-one inference, a single serialized normalization/feature/classifier pipeline, and inspectable coefficients.

## 19. Limitations

- Language detection is uncertain for short, mixed, or named-entity-heavy text.
- Sparse n-grams do not deeply model negation scope, irony, or long-range composition.
- Negative has limited raw and held-out support.
- The supplied labels and their construction were not independently verified.
- No temporal/source fields exist for drift or lineage analysis.

## 20. Sensible production improvements

Collect adjudicated Dutch labels, monitor language/label drift, calibrate and threshold behavior against business costs, add safe model registry promotion, and compare a compact Dutch transformer only after establishing a larger trustworthy benchmark.
