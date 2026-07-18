# Sentiment model comparison after the DeepSeek experiment

## Technical summary

DeepSeek V4 Flash with the frozen 24-shot prompt is the strongest measured classifier on the 960-row held-out set, but it should not automatically replace the local TF-IDF model. It raises held-out Accuracy from 0.6531 to 0.7208 (+6.77 percentage points) and Macro-F1 from 0.6379 to 0.7506 (+11.27 points). All 960 held-out responses were valid, and every predeclared quality gate passed.

The improvement is statistically credible on this split: a paired 2,000-sample bootstrap placed the Accuracy gain at +3.23 to +10.52 points and the Macro-F1 gain at +6.77 to +16.10 points (95% percentile intervals). DeepSeek uniquely corrected 201 rows that TF-IDF missed, while TF-IDF uniquely corrected 136; exact McNemar p=0.00047.

Recommendation: keep TF-IDF as the default low-latency, private and deterministic production path, and advance DeepSeek to a separate production-design review or optional fallback experiment. Before any replacement decision, collect a new blind test set and measure true online latency, privacy constraints, failure handling and recurring cost.

## DeepSeek materially improves minority-class coverage

| Model | Evaluation | Accuracy | Macro-F1 | Negative P/R/F1 | Average F1 | Dutch Macro-F1 | English Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash, 24-shot | Frozen holdout, n=960 | 0.7208 | 0.7506 | 0.7746 / 0.9167 / 0.8397 | 0.6598 | 0.7657 | 0.5669 |
| Official TF-IDF + LR | Frozen holdout, n=960 | 0.6531 | 0.6379 | 0.7209 / 0.5167 / 0.6019 | 0.6603 | 0.6381 | 0.3289 |

DeepSeek's main gain is not from the already common Average class. Average F1 is essentially unchanged (0.6598 versus 0.6603), while Negative recall rises from 51.67% (31/60) to 91.67% (55/60) without sacrificing Negative precision. It also shifts the model away from TF-IDF's English Average-class collapse: English Macro-F1 rises from 0.3289 to 0.5669, though this slice contains only 97 reviews and only two English Negative examples.

## Earlier experimental candidates remain non-promotable

| Candidate | Evidence basis | Accuracy | Macro-F1 | Negative F1 | Decision |
|---|---|---:|---:|---:|---|
| LogisticRegression C=0.5 | 5-fold training CV/OOF | 0.6722 | 0.6536 | 0.6128 | Do not promote; improvement gate failed |
| LinearSVC C=0.5 | 5-fold training CV/OOF | 0.6623 | 0.6325 | 0.5668 | Do not promote |
| Frozen Dutch RoBERT embeddings + LR | 5-fold training CV/OOF | 0.5683 | 0.5165 | 0.3672 | Do not promote |

These rows are useful directional comparisons, not a single leaderboard: they were selected and measured with training-partition OOF/CV, while DeepSeek and the official model have held-out results. No earlier candidate justified consuming the frozen holdout.

## Scope, definitions and validation

- Frozen heldout: the same deduplicated, language-by-label-stratified 960 rows used for the official model, with 450 Positive, 450 Average and 60 Negative reviews.
- Macro-F1: the unweighted mean of class F1 values, used as the primary metric because Negative is only 6.25% of the held-out set.
- DeepSeek prompt: 24 deterministic training examples, eight per label and split into six Dutch plus two English examples per label. Those 24 rows were excluded from the 3,814-row training-partition diagnostic.
- Independent QA: the prediction CSV contained exactly 4,774 unique source rows and review hashes, 3,814 training diagnostics plus 960 held-out predictions, with zero invalid responses. Recomputed held-out metrics and confusion matrix exactly match the saved JSON.

## Operational evidence and limitations

- The run made 4,774 successful first-attempt calls and processed 18.02M tokens. Estimated experiment cost was $0.3558 at the price assumptions saved on 18 July 2026, about $0.0745 per 1,000 reviews for this batch. Future prices and prompt-cache behavior can change.
- The original latency timer included time waiting behind the 20-request concurrency limit. Its reported p50/p95 values are invalid as per-request latency and must not be cited. Cache file timestamps span approximately 269 seconds, which only supports a batch-throughput observation of roughly 17.7 reviews/second, not an online latency SLA.
- DeepSeek returns hard labels without calibrated class probabilities, so it cannot replace the current model's confidence, calibration and threshold behavior without a separate design.
- Sending reviews to an external provider creates privacy, availability, vendor-drift and cost dependencies. The public model name may remain stable while behavior changes.
- The held-out set has now been observed for both TF-IDF and DeepSeek. Any further prompt or model choice informed by these results requires a newly collected blind test set for an unbiased final claim.

## Recommended next steps

1. Keep the current TF-IDF artifact unchanged as the default production baseline.
2. Freeze the DeepSeek prompt, model name and parsing rules; do not tune them on the current held-out errors.
3. Collect and label a new blind test set, deliberately increasing Dutch and English Negative coverage.
4. Run a production pilot that records true request-only p50/p95 latency, timeout rate, retry rate, provider drift and cost per 1,000 reviews.
5. Decide between three architectures after the pilot: local-only TF-IDF, DeepSeek-only, or a selective fallback where the local model handles confident cases and DeepSeek handles ambiguous cases.

## Further questions

- Does DeepSeek retain its Negative recall on newly collected data rather than this one frozen split?
- Can a selective fallback capture most of the quality gain at lower cost and lower data exposure?
- What privacy policy applies to sending raw review text to an external provider?
