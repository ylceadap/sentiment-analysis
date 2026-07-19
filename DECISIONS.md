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
- **Limitations:** the original machine still cannot run Docker locally.
- **Follow-up verification:** GitHub Actions run 1 later built the serving image, started the container, and passed its `/health` check on Linux.

## D007 — Use a local SQLite MLflow backend

- **Alternatives considered:** legacy MLflow directory file store; local SQLite backend; custom experiment CSV only.
- **Decision:** use `sqlite:///mlflow.db` for live tracking and export a compact comparison CSV to the repository.
- **Reasoning:** installed MLflow 3.14 places the legacy directory store in maintenance mode and refuses new writes by default; SQLite is local, supported, inspectable, and keeps the same one-command UI workflow.
- **Consequences:** `mlflow.db` is intentionally ignored while portable comparison and evidence artifacts are tracked.
- **Limitations:** evaluators do not receive the local run database unless it is explicitly packaged; the exported table and metadata preserve key results.

## D008 — Filter confident non-Dutch training rows and allow short ambiguous API input (superseded by D012)

- **Alternatives considered:** accept all rows; reject every uncertain input; use confidence/margin thresholds plus a short-text ambiguity state.
- **Decision:** train only on confident Dutch candidates. The API returns HTTP 422 for confident non-Dutch text but permits text shorter than 20 characters as `ambiguous`.
- **Reasoning:** this follows the Dutch-only requirement without pretending language ID is perfect on inputs such as “Goed”.
- **Consequences:** 485 English candidates are removed; short non-Dutch text may occasionally reach sentiment classification.
- **Limitations:** mixed-language and named-entity-heavy text can still be misidentified.

## D009 — Retain explicit ratings in the final pipeline

- **Alternatives considered:** always retain ratings; always mask them; select using a paired experiment.
- **Decision:** retain ratings because the final unified-data paired candidate achieved CV macro-F1 0.6472 versus 0.6459 when masked.
- **Reasoning:** the 0.0014 difference is small relative to fold standard deviations, so ratings are documented as a leakage risk rather than treated as the main performance source.
- **Consequences:** legitimate rating language remains available to the model and the selected pipeline follows the predefined metric.
- **Limitations:** source-label construction is unknown, so direct label leakage cannot be ruled out.

## D010 — Use a single transformed feature vector per API prediction

- **Alternatives considered:** keep independent `predict`, `predict_proba`, and `explain` calls; cache only explanations; add one application-facing inference operation.
- **Decision:** add `SentimentModel.infer`, which normalizes/vectorizes once and derives the label, native probabilities, and optional explanation from the same sparse vector. Cache immutable feature names after the first explanation.
- **Reasoning:** the previous service repeated the most expensive transformation two to four times per request and regenerated roughly 90,000 feature names for every explanation.
- **Consequences:** faster inference, internally consistent outputs, and a small lazy memory cache after explanations are used.
- **Limitations:** the first explanation still pays one-time feature-name construction cost.

## D011 — Separate serving dependencies from training dependencies

- **Alternatives considered:** install the full analytics stack in Docker; maintain a duplicate requirements file; use optional dependency groups.
- **Decision:** keep FastAPI, Lingua, sklearn, and serialization libraries in core dependencies; move MLflow, pandas, PyYAML, and tabulate to the `train` extra. Development installation uses `.[train,dev]`, while Docker installs core only.
- **Reasoning:** the existing local environment was 837 MB and unnecessarily pulled MLflow, pyarrow, matplotlib, and database tooling into the serving image.
- **Consequences:** smaller and lower-risk serving images without duplicating version constraints.
- **Limitations:** audit/training commands require the documented `train` extra.

## D012 — Train one shared model on all supplied Dutch and English rows

- **Alternatives considered:** keep filtering English; train separate language models; train one model on all supplied rows with language-aware evaluation.
- **Decision:** supersede D008's training filter. Retain every deduplicated supplied row, use one shared feature/model pipeline, jointly stratify language and label for holdout/CV, and accept both Dutch and English at inference. English responses include a reliability warning.
- **Reasoning:** the supplied dataset contains 485 consistently detected English reviews, Dutch and English share useful lexical/character patterns, and a single mixed model follows the user's explicit scope without pretending the small English segment supports a separate model.
- **Consequences:** the training population grows from 4,313 to 4,798 deduplicated reviews; overall and per-language held-out metrics are both required; English requests no longer receive HTTP 422.
- **Limitations:** English labels are highly imbalanced—only 10 raw English Negative rows—so English and especially English-Negative metrics remain descriptive rather than conclusive. Confidently detected languages other than Dutch or English remain unsupported.

## D013 — Use one production champion and evidence-only research runs

- **Alternatives considered:** label every registered candidate as champion; register reports as models; separate deployable artifacts from research evidence.
- **Decision:** keep only `sentiment-production@champion`. Mark other artifacts as baseline, benchmark, challenger, research-only, or external advisor, and archive branch results in a separate evidence experiment.
- **Reasoning:** a Registry alias must communicate deployment authority, while a strong metric or report does not prove that a self-contained deployable model exists.
- **Consequences:** ordinal and Jina evidence is searchable in MLflow without being misrepresented as production-ready.
- **Limitations:** local SQLite and `mlruns/` remain outside Git and require a separate backup.

## D014 — Align supported Python and all public versions with verified evidence

- **Alternatives considered:** claim untested Python 3.12 support; keep duplicated `0.1.0` strings; use one package version source and a verified lock.
- **Decision:** support Python 3.11, derive build and API versions from `dutch_sentiment.__version__`, retain compatible dependency groups, and add an exact verified Python 3.11 environment lock.
- **Reasoning:** support and reproducibility claims should reflect environments that were actually installed and tested.
- **Consequences:** editable installation, FastAPI metadata, Docker, and training model versions agree.
- **Limitations:** the lock is platform-specific to the verified macOS x86_64 environment; other platforms resolve from `pyproject.toml`.

## D015 — Share experiment invariants without merging research policies

- **Alternatives considered:** keep two monolithic scripts; merge all experiment behavior; extract stable mechanics while leaving model-specific selection visible.
- **Decision:** share frozen split preparation, hashes, probability alignment, language/fold summaries, promotion gates, embedding cache mechanics, and ordinal equations. Keep each experiment's candidate construction and ranking policy in its own orchestrator.
- **Reasoning:** the shared mechanics must not drift, but hiding model-specific decisions in an overly generic framework would make leakage and selection logic harder to audit.
- **Consequences:** the embedding and Jina ordinal orchestrators are smaller and their common invariants have direct tests.
- **Limitations:** report composition and model-specific fit loops remain intentionally separate.

## D016 — Separate models by package and configuration, not permanent Git branches

- **Alternatives considered:** retain one branch per model; merge every experiment branch wholesale;
  keep one stable code line with archived experiment tags and MLflow evidence.
- **Decision:** keep `main` as the only long-lived code branch. Put deployable algorithms in
  `models/`, research orchestration in `experiments/`, and lifecycle-specific settings in
  `configs/models/`. Tag completed experiment tips before deleting their branches.
- **Reasoning:** branch ancestry had mixed model families and made a branch name look like a model
  identity. MLflow already provides the correct run, artifact, and lifecycle boundary.
- **Consequences:** models can be compared from one checkout, the service has one stable contract,
  and old experiment code remains recoverable through remote archive tags.
- **Limitations:** the local MLflow database still needs an independent backup or shared backend.

## D017 — Export the champion to a verified serving artifact

- **Alternatives considered:** query MLflow on every API startup; serve an unverified checked-in pickle;
  export the champion's exact source-run artifact and bind it to a tracked release manifest.
- **Decision:** keep the serving image independent of MLflow and verify `model.joblib`, metadata,
  `model_release.json`, Registry alias, and source-run copies before release.
- **Reasoning:** this preserves a small runtime while preventing Registry and Docker from drifting.
- **Consequences:** CI checks the tracked release; local promotion additionally requires MLflow.
- **Limitations:** the local Registry remains the source of promotion authority and needs its external backup.

## D018 — Keep research inference out of the production UI

- **Alternatives considered:** expose all Registry entries in one selector; create a research UI; keep one
  formal model plus a visibly advisory external comparison.
- **Decision:** the production UI runs only the verified TF-IDF champion and optional versioned
  `zero-shot-advisor-v1`. It may display the frozen five-model ranking as read-only evidence, but
  Jina, TF-IDF Ordinal, and historical DeepSeek 24-shot are never live inference choices. RobBERT,
  benchmark, and ablation models remain MLflow-only test evidence.
- **Reasoning:** model visibility must not imply production approval. The historical DeepSeek 24-shot
  result is also kept separate because the runtime prompt is not that evaluated configuration.
- **Consequences:** the inference contract stays simple and honest while the requested final ranking
  is visible without implying deployment approval.
- **Limitations:** adding an approved challenger to the UI requires a deliberate new endpoint and review.
