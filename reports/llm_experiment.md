# Direct LLM sentiment experiment

## Decision

**Eligible for a separate production-design review.** The experiment used `deepseek-v4-flash` in non-thinking, 24-shot JSON mode. It did not train model weights.

Held-out Macro-F1: **0.7506** versus **0.6379** for the official TF-IDF model. Held-out accuracy: 0.7208 versus 0.6531. Negative precision/recall: 0.7746/0.9167.

## Promotion gates

- PASS — `macro_f1`
- PASS — `negative_precision`
- PASS — `negative_recall`
- PASS — `invalid_response_rate`
- PASS — `accuracy`

## Evidence

- Prompt examples: 24 (8 per class; 6 Dutch and 2 English per class).
- Train-partition evaluation rows: 3814 (prompt examples excluded).
- Untouched held-out evaluation rows: 960.
- Invalid held-out responses after retries: 0.
- API calls/cache hits: 4774/0.
- Estimated API cost at the prices recorded on 2026-07-18: $0.3558.
- API request latency is unavailable for this run because the original timer included concurrency queue time. Cache timestamps show the 4,774 responses were written over approximately 269 seconds, which is a batch-throughput observation rather than per-request latency.

## Limitations

- Provider model behavior can change even when the public model name is stable.
- Reviews are sent to an external API, adding privacy, availability, latency, and cost concerns.
- The 24 examples were deterministically selected from the training partition; those rows are excluded from training-partition metrics.
- The held-out set was evaluated once after the prompt and model configuration were frozen.
- LLM labels have no calibrated probability, so probability metrics and threshold tuning are unavailable.
