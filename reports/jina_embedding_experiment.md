# Jina v3 classification-embedding experiment

## Decision

**Advance to a separately budgeted GPU fine-tune and a new blind test.** The official TF-IDF model and held-out test were not changed or evaluated.

Best experimental candidate: `jina_v3_classification__c2.0__negative_7__threshold_argmax`. CV Macro-F1 is 0.7108, versus 0.6472 for the official baseline. Negative precision/recall are 0.6196/0.8958, versus 0.7333/0.5042.

## Promotion gates

- PASS — `macro_f1`
- PASS — `negative_precision`
- PASS — `negative_recall`
- PASS — `accuracy`
- PASS — `stability`

## Method

- Recreated and hash-verified the frozen 3,838-row training split and 960-row holdout split.
- Encoded Dutch and English training reviews together using one frozen, revision-pinned Jina v3 classification encoder; this run used `cuda` on a Colab T4.
- Used the same five stratified folds for all candidates; labels never fitted the encoders.
- Selected C, class weight, and Negative threshold only from out-of-fold predictions.
- Did not compute holdout metrics, replace `artifacts/model.joblib`, or train an English-only model.

## Top OOF configurations

| name                                                       |   cv_macro_f1_mean |   cv_macro_f1_std |   oof_accuracy |   negative_precision |   negative_recall |   negative_f1 |   dutch_macro_f1 |   english_macro_f1 |
|:-----------------------------------------------------------|-------------------:|------------------:|---------------:|---------------------:|------------------:|--------------:|-----------------:|-------------------:|
| jina_v3_classification__c2.0__negative_7__threshold_argmax |             0.7108 |            0.0122 |         0.7024 |               0.6196 |            0.8958 |        0.7325 |           0.7101 |             0.6615 |
| jina_v3_classification__c1.0__negative_7__threshold_argmax |             0.7098 |            0.0158 |         0.704  |               0.6022 |            0.9083 |        0.7243 |           0.7099 |             0.6536 |
| jina_v3_classification__c2.0__balanced__threshold_argmax   |             0.7072 |            0.0115 |         0.7011 |               0.6006 |            0.9083 |        0.7231 |           0.7068 |             0.6566 |
| jina_v3_classification__c1.0__balanced__threshold_argmax   |             0.7071 |            0.0114 |         0.7035 |               0.5882 |            0.9167 |        0.7166 |           0.7087 |             0.6414 |
| jina_v3_classification__c0.5__negative_7__threshold_argmax |             0.7071 |            0.0119 |         0.7038 |               0.5887 |            0.9125 |        0.7157 |           0.7091 |             0.6393 |
| jina_v3_classification__c2.0__negative_7__threshold_0.4    |             0.7051 |            0.0128 |         0.6988 |               0.5994 |            0.9042 |        0.7209 |           0.7042 |             0.6541 |
| jina_v3_classification__c0.5__negative_7__threshold_0.4    |             0.7045 |            0.0111 |         0.7014 |               0.5796 |            0.925  |        0.7127 |           0.7078 |             0.6205 |
| jina_v3_classification__c1.0__balanced__threshold_0.4      |             0.7042 |            0.0092 |         0.7014 |               0.5781 |            0.925  |        0.7115 |           0.707  |             0.6226 |
| jina_v3_classification__c1.0__negative_7__threshold_0.4    |             0.704  |            0.0126 |         0.7004 |               0.5856 |            0.9125 |        0.7134 |           0.7059 |             0.628  |
| jina_v3_classification__c2.0__balanced__threshold_0.4      |             0.7035 |            0.0114 |         0.6983 |               0.5903 |            0.9125 |        0.7169 |           0.7037 |             0.6422 |

## Reproducibility and limitations

Exact model revisions, split hashes, search grid, cache paths, and gates are in `configs/models/jina_logreg.yaml`. Embeddings and model downloads are local-only. English slice results are directional: training has 388 English rows and only 8 English Negative rows. A future final decision requires a newly collected blind test.

License note: Jina Embeddings v3 is CC-BY-NC-4.0. This branch is a non-commercial research experiment and is not automatically eligible for production promotion even if the metric gates pass.

## Independent validation

The exported CSV contains one official baseline and 30 unique candidates from the declared 3 × 2 × 5
grid. Split hashes reproduce 3,838 training rows and 960 test rows; the experiment itself did not
evaluate the test partition or replace the official model. Recomputing the five promotion gates
reproduces five passes, and the selected candidate is unique.

Compared with the official OOF baseline, the selected candidate changes Macro-F1 from 0.6472 to
0.7108, accuracy from 0.6699 to 0.7024, and Negative F1 from 0.5975 to 0.7325. Negative precision
falls from 0.7333 to 0.6196 while recall rises from 0.5042 to 0.8958.

The assessment is **share with caveats**: these are training-set out-of-fold results, English has
only 388 training rows and eight Negative examples, and a genuinely new blind set would be required
for promotion. The pinned encoder revision is
`ab036b023d30b4d1138c4c3bfa9f0c445ab455d6`; the observed remote-code dependency is
`845308d0fd72a8406a3e378450e1a09522790419`. `trust_remote_code` executes third-party code and must
be reviewed before future runs.
