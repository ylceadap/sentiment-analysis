"""Generate the evidence-backed model report from machine-readable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _format_metric(value: object) -> str:
    return f"{float(value):.4f}" if value is not None else "n/a"


def build_model_report(artifact_dir: str | Path, output: str | Path) -> None:
    """Build a durable Markdown report without inventing missing evidence."""
    artifacts = Path(artifact_dir)
    audit = _read_json(artifacts / "data_audit.json")
    metrics = _read_json(artifacts / "final_metrics.json")
    metadata = _read_json(artifacts / "model_metadata.json")
    split = _read_json(artifacts / "split_metadata.json")
    benchmark = _read_json(artifacts / "benchmark.json")
    comparison_path = artifacts / "experiment_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.is_file() else pd.DataFrame()
    comparison_md = (
        comparison.round(4).to_markdown(index=False) if not comparison.empty else "Pending."
    )
    per_class = metrics.get("per_class", {})
    per_class_rows = [
        {
            "label": label,
            "precision": values.get("precision"),
            "recall": values.get("recall"),
            "f1": values.get("f1-score"),
            "support": values.get("support"),
        }
        for label, values in per_class.items()
    ]
    per_class_md = (
        pd.DataFrame(per_class_rows).round(4).to_markdown(index=False)
        if per_class_rows
        else "Pending."
    )
    language_metric_rows = []
    for language_name, values in metrics.get("by_language", {}).items():
        language_per_class = values.get("per_class", {})
        language_metric_rows.append(
            {
                "detected_language": language_name,
                "support": sum(
                    float(item.get("support", 0)) for item in language_per_class.values()
                ),
                "accuracy": values.get("accuracy"),
                "balanced_accuracy": values.get("balanced_accuracy"),
                "macro_f1": values.get("macro_f1"),
                "negative_f1": language_per_class.get("Negative", {}).get("f1-score"),
                "negative_support": language_per_class.get("Negative", {}).get("support"),
                "log_loss": values.get("log_loss"),
            }
        )
    language_metrics_md = (
        pd.DataFrame(language_metric_rows).round(4).to_markdown(index=False)
        if language_metric_rows
        else "Pending."
    )
    latency_rows = []
    for name, values in benchmark.get("measurements", {}).items():
        latency_rows.append(
            {
                "path": name,
                "iterations": values.get("iterations"),
                "mean_ms": values.get("mean_ms"),
                "p50_ms": values.get("p50_ms"),
                "p95_ms": values.get("p95_ms"),
                "max_ms": values.get("max_ms"),
            }
        )
    latency_md = (
        pd.DataFrame(latency_rows).round(3).to_markdown(index=False)
        if latency_rows
        else "Not measured yet."
    )
    confusion = metrics.get("confusion_matrix")
    confusion_text = (
        pd.DataFrame(
            confusion,
            index=metrics.get("label_order"),
            columns=metrics.get("label_order"),
        ).to_markdown()
        if confusion
        else "Pending."
    )
    rating_rows = comparison.loc[
        comparison.get("feature_kind", pd.Series(dtype=str)).eq("combined")
        & comparison.get("class_weight", pd.Series(dtype=str)).fillna("none").eq("balanced")
    ]
    rating_md = (
        rating_rows.round(4).to_markdown(index=False) if not rating_rows.empty else "Pending."
    )
    error_path = artifacts / "error_analysis.csv"
    errors = pd.read_csv(error_path) if error_path.is_file() else pd.DataFrame()
    error_md = (
        errors.head(12).to_markdown(index=False) if not errors.empty else "No errors recorded."
    )
    language = audit.get("language", {})
    selected = metadata.get("experiment", {})

    report = f"""# Model Report

## 1. Executive summary

The final system uses `{selected.get("name", "pending")}`. Selection is based on stratified cross-validation macro-F1 over the training partition, with Negative-class behavior, latency, size, probability support, and explanation feasibility treated as engineering constraints. The held-out test set is used only after selection.

## 2. Dataset description

- Raw rows: {audit.get("schema", {}).get("rows", "n/a")}
- Raw SHA-256: `{audit.get("source", {}).get("sha256", "n/a")}`
- Original labels: {json.dumps(audit.get("labels", {}), ensure_ascii=False)}
- Text is label-ordered, so sequential splitting is invalid.

## 3. Language composition and unified training policy

- Status counts: {json.dumps(language.get("status_counts", {}), ensure_ascii=False)}
- Status by label: {json.dumps(language.get("status_by_label", {}), ensure_ascii=False)}
- Every deduplicated Dutch and English row is retained in one shared model; no language-specific model is trained.
- Holdout and CV splits jointly stratify detected language and label.
- Detector output is not a gold annotation; manual bounded examples are in `reports/data_audit.md`.

## 4. Data-quality findings

- Exact duplicate extras: {audit.get("duplicates", {}).get("exact_extra_rows", "n/a")}
- Normalized conflicting groups: {audit.get("duplicates", {}).get("normalized_conflicting_groups", "n/a")}
- HTML breaks: {audit.get("artifacts", {}).get("html_break", "n/a")}; zero-width characters: {audit.get("artifacts", {}).get("zero_width", "n/a")}; mojibake candidates: {audit.get("artifacts", {}).get("mojibake", "n/a")}.

## 5. Leakage risks

The pipeline prevents normalized duplicate overlap, fits all vectorizers inside CV folds, never uses labels during language detection, and compares explicit ratings retained versus replaced by a neutral token. Ratings remain legitimate review content but may directly encode how source labels were assigned.

## 6. Split methodology

{json.dumps(split, ensure_ascii=False, indent=2)}

## 7–8. Experiment and cross-validation results

{comparison_md}

## 9. Final held-out test metrics

- Accuracy: {_format_metric(metrics.get("accuracy"))}
- Balanced accuracy: {_format_metric(metrics.get("balanced_accuracy"))}
- Macro precision: {_format_metric(metrics.get("macro_precision"))}
- Macro recall: {_format_metric(metrics.get("macro_recall"))}
- Macro-F1: {_format_metric(metrics.get("macro_f1"))}
- Weighted F1: {_format_metric(metrics.get("weighted_f1"))}
- Log loss: {_format_metric(metrics.get("log_loss"))}
- Multiclass Brier score: {_format_metric(metrics.get("multiclass_brier_score"))}
- Expected calibration error (10 bins): {_format_metric(metrics.get("expected_calibration_error_10_bin"))}
- Mean prediction confidence: {_format_metric(metrics.get("mean_prediction_confidence"))}

These probability metrics are descriptive evidence on the held-out set. Logistic Regression supplies native probabilities, but no separate calibration model was fitted; the calibration estimate is not an operational guarantee.

### Held-out metrics by detected language

{language_metrics_md}

Language slices evaluate the same shared model and are not separately trained models. English results require extra caution because the source has only 485 English rows and 10 English Negative rows; the held-out English-Negative support is especially small.

## 10. Per-class metrics

{per_class_md}

## 11. Confusion matrix

Rows are true labels and columns are predicted labels.

{confusion_text}

## 12. Negative-class analysis

Negative recall is {_format_metric(per_class.get("Negative", {}).get("recall"))} and Negative F1 is {_format_metric(per_class.get("Negative", {}).get("f1-score"))}. This class has the smallest support, so point estimates should be interpreted with more uncertainty than Positive or Average results.

## 13. Rating-leakage experiment

{rating_md}

The paired rows differ only in rating masking. The final choice follows CV macro-F1 and Negative-class evidence rather than automatically retaining the higher-leakage signal.

## 14. Error analysis

Excerpts are deliberately capped and whitespace-normalized.

{error_md}

Observed failures may combine ambiguous sentiment, label noise, sparse Negative evidence, cross-language imbalance, and bag-of-ngrams limitations. Feature contributions do not establish causal reasons.

## 15–16. Inference latency, size, and load time

{latency_md}

- Model artifact bytes: {metadata.get("model_size_bytes", "pending")}
- Model SHA-256: `{metadata.get("model_sha256", "pending")}`
- Language-detector constructor (ms): {benchmark.get("detector_initialization_ms", "not measured")}
- Cold first language inference/model load (ms): {benchmark.get("cold_detector_first_inference_ms", "not measured")}
- Cold load time (ms): {benchmark.get("cold_model_load_ms", "not measured")}

## 17. Prediction explanation

The API can return active word n-grams supporting/opposing the selected class and a separate technical list of character n-grams. Contributions are local linear associations, not causal or semantic explanations.

## 18. Final model-selection reasoning

Selected specification: `{json.dumps(selected, ensure_ascii=False)}`. It is a CPU-friendly sparse linear model with native multiclass probabilities, fast batch-size-one inference, a single serialized normalization/feature/classifier pipeline, and inspectable coefficients.

## 19. Limitations

- Language detection is uncertain for short, mixed, or named-entity-heavy text.
- English predictions are supported but less reliable because English supervision is limited and highly class-imbalanced.
- Sparse n-grams do not deeply model negation scope, irony, or long-range composition.
- Negative has limited raw and held-out support.
- The supplied labels and their construction were not independently verified.
- No temporal/source fields exist for drift or lineage analysis.

## 20. Sensible production improvements

Collect adjudicated Dutch and English labels, monitor metrics and drift by language/label, calibrate and threshold behavior against business costs, add safe model registry promotion, and compare a compact multilingual transformer only after establishing a larger trustworthy benchmark.
"""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
