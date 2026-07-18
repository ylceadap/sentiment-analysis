# Decision Log

## D001 — Treat the PDF as authoritative and the expanded brief as execution guidance

- **Alternatives considered:** implement only the expanded brief; implement only the PDF; reconcile both.
- **Decision:** the supplied PDF defines acceptance requirements. The expanded brief supplies stronger engineering and verification practices where they do not conflict.
- **Reasoning:** this prevents optional guidance from being misreported as an original requirement while still producing a credible take-home submission.
- **Consequences:** requirement traceability distinguishes required items from bonuses and engineering additions.
- **Limitations:** evaluator preferences not stated in the PDF remain assumptions.

## D002 — Use a classical sparse-text model before considering transformers

- **Alternatives considered:** Dutch transformer fine-tuning; classical TF-IDF linear models; heuristic sentiment rules.
- **Decision:** compare Dummy, word, character, and combined TF-IDF linear baselines first; do not add a transformer unless all required engineering work is verified and it adds useful evidence.
- **Reasoning:** the dataset is small and imbalanced, while the challenge emphasizes approach, experiments, latency, software design, and explainability.
- **Consequences:** training is CPU-friendly, inference is fast, and linear contributions can support local explanations.
- **Limitations:** bag-of-ngrams will be weaker on compositional meaning, sarcasm, and long-range context.

## D003 — Use macro-F1 for model selection

- **Alternatives considered:** accuracy, weighted F1, macro-F1, Negative-only recall.
- **Decision:** macro-F1 is the primary CV selection metric; balanced accuracy, per-class metrics, and especially Negative recall/F1 are mandatory supporting evidence.
- **Reasoning:** the raw label distribution is strongly imbalanced, so accuracy and weighted averages can obscure poor Negative performance.
- **Consequences:** class-balanced candidates remain competitive even if raw accuracy falls slightly.
- **Limitations:** macro-F1 weights all classes equally but does not encode a business cost matrix.

## D004 — Preserve the source and prevent normalized duplicate leakage

- **Alternatives considered:** mutate the CSV; randomly split rows; deduplicate/group before splitting.
- **Decision:** never modify the source CSV. Normalize deterministically, remove or group normalized duplicates before a shuffled stratified split, and persist split metadata.
- **Reasoning:** the CSV is label-ordered and contains duplicates; sequential or ungrouped row splits could invalidate evaluation.
- **Consequences:** sample counts may fall slightly and split logic is more careful.
- **Limitations:** near-duplicates that survive normalization may still cross splits.

## D005 — Use Git plus lightweight file/hash provenance without DVC

- **Alternatives considered:** Git only; Git plus DVC; MLflow plus hashes/config/metadata without Git.
- **Decision:** initialize Git because the user explicitly requested it. Combine focused commits with MLflow, raw/model hashes, configuration, split metadata, package versions, and model metadata; do not add DVC.
- **Reasoning:** Git now provides useful code history, while DVC would remain decorative for one immutable supplied CSV.
- **Consequences:** verified phases can be committed independently and the final model metadata can include a real commit when available.
- **Limitations:** the raw CSV is tracked for challenge reproducibility, so repository size is larger than a code-only project.

## D006 — Docker runtime verification is environment-dependent

- **Alternatives considered:** claim Docker compatibility from inspection; install Docker; supply and statically review a Dockerfile.
- **Decision:** create and inspect a serving image definition, but report runtime verification as unavailable unless a Docker executable becomes available.
- **Reasoning:** Docker is not installed in the current environment and successful container execution must not be fabricated.
- **Consequences:** local API verification remains required; Docker commands will still be documented.
- **Limitations:** build-time dependency or runtime issues cannot be ruled out without an actual engine.
