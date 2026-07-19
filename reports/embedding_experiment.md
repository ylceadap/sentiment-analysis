# Frozen sentence-embedding experiment

## Decision

**Stop here: no candidate cleared every predeclared OOF promotion gate.** The official TF-IDF model and untouched held-out test were not changed or evaluated.

Selected experimental candidate: `dutch_robbert__c0.25__negative_7__threshold_0.4`. Its CV Macro-F1 is 0.5165, versus 0.6472 for the reproduced official baseline. Negative precision/recall are 0.2615/0.6167, versus 0.7333/0.5042.

## Promotion gates

- FAIL — `macro_f1`
- FAIL — `negative_precision`
- PASS — `negative_recall`
- FAIL — `accuracy`
- PASS — `stability`

## Method

- Recreated and hash-verified the frozen 3,838-row training split and 960-row holdout split.
- Encoded Dutch and English training reviews together with two revision-pinned, frozen CPU encoders.
- Used the same five stratified folds for every candidate; labels were never used to fit the encoders.
- Selected LogisticRegression C, class weight, and Negative threshold only from out-of-fold predictions.
- Did not read holdout labels for metrics, tune on the holdout, replace `artifacts/model.joblib`, or create a separate English model.

## Top OOF configurations

| name                                                  |   cv_macro_f1_mean |   cv_macro_f1_std |   oof_accuracy |   negative_precision |   negative_recall |   negative_f1 |   dutch_macro_f1 |   english_macro_f1 |
|:------------------------------------------------------|-------------------:|------------------:|---------------:|---------------------:|------------------:|--------------:|-----------------:|-------------------:|
| official_tfidf_baseline                               |             0.6472 |            0.0269 |         0.6699 |               0.7333 |            0.5042 |        0.5975 |           0.6524 |             0.2828 |
| dutch_robbert__c0.25__negative_7__threshold_0.4       |             0.5165 |            0.0141 |         0.5683 |               0.2615 |            0.6167 |        0.3672 |           0.5144 |             0.3884 |
| dutch_robbert__c0.5__negative_7__threshold_0.4        |             0.5135 |            0.014  |         0.5667 |               0.2571 |            0.6042 |        0.3607 |           0.5118 |             0.3895 |
| dutch_robbert__c0.5__balanced__threshold_0.4          |             0.513  |            0.0149 |         0.5633 |               0.2541 |            0.6458 |        0.3647 |           0.5107 |             0.3902 |
| multilingual_minilm__c0.75__negative_7__threshold_0.4 |             0.5111 |            0.0118 |         0.5584 |               0.2539 |            0.675  |        0.369  |           0.5098 |             0.4675 |
| dutch_robbert__c0.25__balanced__threshold_0.4         |             0.5106 |            0.0151 |         0.562  |               0.2508 |            0.6333 |        0.3593 |           0.5078 |             0.3908 |
| multilingual_minilm__c0.25__negative_7__threshold_0.4 |             0.51   |            0.0118 |         0.5545 |               0.2612 |            0.6792 |        0.3773 |           0.5095 |             0.4623 |
| dutch_robbert__c0.75__negative_7__threshold_0.4       |             0.5097 |            0.0185 |         0.5628 |               0.2549 |            0.5958 |        0.3571 |           0.5091 |             0.378  |
| dutch_robbert__c0.75__negative_7__threshold_argmax    |             0.5097 |            0.0148 |         0.5618 |               0.25   |            0.625  |        0.3571 |           0.5082 |             0.3823 |
| multilingual_minilm__c1.0__negative_7__threshold_0.4  |             0.5095 |            0.0116 |         0.5591 |               0.2488 |            0.6583 |        0.3611 |           0.5084 |             0.4676 |

## Reproducibility and limitations

The exact model revisions, split hashes, hyperparameter grid, cache locations, and gates are in `configs/embedding_experiment.yaml`. Cached embeddings are local-only and are not committed. English slice results are directional because the training split contains only 388 English rows and 8 English Negative rows. Passing these OOF gates would justify—not prove—a later GPU fine-tune; final promotion still requires a newly collected blind test.

Decision metadata: `{"advance_to_gpu_finetuning": false, "data": {"heldout_normalized_sha256": "b76afd73eaeed79bf61903ab8475a08c4565cf523a55e8ae34448d2330e00cbb", "heldout_rows": 960, "raw_sha256": "2788b987e2c9fa4fd6459a1798a6e7d1dd63ddb5618c10907712d39042cc4be2", "train_normalized_sha256": "aa986ef1d8f35ebf232a015fd0e61d3affb1efbe4a589ec4f8653ae01e8ab7c9", "train_rows": 3838}, "environment": {"platform": "macOS-26.0-x86_64-i386-64bit", "python": "3.11.7", "scikit_learn": "1.9.0"}, "gate_checks": {"accuracy": false, "macro_f1": false, "negative_precision": false, "negative_recall": true, "stability": true}, "heldout_evaluated": false, "next_step": "Collect and review more Dutch Negative examples before GPU fine-tuning", "official_model_replaced": false, "selected_candidate": "dutch_robbert__c0.25__negative_7__threshold_0.4"}`
