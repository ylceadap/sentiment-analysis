"""Evaluate the frozen ordinal candidate and promote it only when every gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from . import __version__
from .config import load_config
from .constants import LABELS, MAX_REVIEW_CHARACTERS
from .data import annotate_review_languages, load_dataset, make_holdout_split, sha256_file
from .language import DutchLanguageDetector
from .metrics import classification_metrics
from .model import build_ordinal_model


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _hash_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _ordinal_metrics_from_confusion(metrics: dict[str, Any]) -> dict[str, float]:
    labels = list(metrics["label_order"])
    y_true: list[str] = []
    y_pred: list[str] = []
    for true_index, row in enumerate(metrics["confusion_matrix"]):
        for predicted_index, count in enumerate(row):
            y_true.extend([labels[true_index]] * int(count))
            y_pred.extend([labels[predicted_index]] * int(count))
    reconstructed = classification_metrics(y_true, y_pred)
    return {
        key: float(reconstructed[key])
        for key in (
            "ordinal_mae",
            "quadratic_weighted_kappa",
            "adjacent_error_rate",
            "severe_error_rate",
        )
    }


def promotion_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    """Apply the pre-declared class-imbalance replacement constraints."""
    candidate_negative = candidate["per_class"]["Negative"]
    baseline_negative = baseline["per_class"]["Negative"]
    return {
        "negative_precision_at_least_0_60": candidate_negative["precision"] >= 0.60,
        "negative_recall_improved": candidate_negative["recall"] > baseline_negative["recall"],
        "negative_f1_not_lower": candidate_negative["f1-score"] >= baseline_negative["f1-score"],
        "macro_f1_not_lower": candidate["macro_f1"] >= baseline["macro_f1"],
        "balanced_accuracy_not_lower": candidate["balanced_accuracy"]
        >= baseline["balanced_accuracy"],
    }


def ordinal_diagnostics(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    """Report ordinal trade-offs without retroactively making them promotion gates."""
    return {
        "ordinal_mae_not_higher": candidate["ordinal_mae"] <= baseline["ordinal_mae"],
        "quadratic_weighted_kappa_not_lower": candidate["quadratic_weighted_kappa"]
        >= baseline["quadratic_weighted_kappa"],
        "severe_error_rate_not_higher": candidate["severe_error_rate"]
        <= baseline["severe_error_rate"],
    }


def _metrics_by_language(
    languages: list[str],
    labels: list[str],
    predictions: list[str],
    probabilities: list[dict[str, float]],
) -> dict[str, dict[str, Any]]:
    slices: dict[str, dict[str, Any]] = {}
    for language in sorted(set(languages)):
        indices = [index for index, value in enumerate(languages) if value == language]
        slices[language] = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
            [probabilities[index] for index in indices],
        )
    return slices


def run_ordinal_promotion(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["random_seed"])
    output_dir = Path(config["output_dir"])
    experiment_dir = Path(config["ordinal_logistic"]["output_dir"])
    selection_path = experiment_dir / "ordinal_logistic_experiment.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("new_blind_evaluation_recommended"):
        raise RuntimeError("The ordinal candidate did not pass the training-only OOF gate")
    selected_name = str(selection["selected_candidate"])
    selected = selection["candidates"][selected_name]
    if selected["kind"] != "two_boundary_crossfit_threshold":
        raise RuntimeError("The frozen candidate is not a cross-fitted threshold model")

    raw = load_dataset(config["data_path"])
    detector = DutchLanguageDetector(**config["language"])
    annotated, _ = annotate_review_languages(raw, detector)
    split = make_holdout_split(
        annotated,
        test_size=float(config["test_size"]),
        random_seed=seed,
        stratify_columns=("detected_language", "Label"),
    )
    train_reviews = split.train["Reviews"].astype(str).tolist()
    train_labels = split.train["Label"].astype(str).tolist()
    test_reviews = split.test["Reviews"].astype(str).tolist()
    test_labels = split.test["Label"].astype(str).tolist()
    test_languages = split.test["detected_language"].fillna("unknown").astype(str).tolist()

    git_commit = _git_commit()
    git_dirty = _git_dirty()
    parameters = selected["parameters"]
    version = f"{__version__}+ordinal.{(git_commit or 'nogit')[:8]}"
    model = build_ordinal_model(
        {**config["model"], "random_seed": seed},
        c_value=float(parameters["C"]),
        calibration_folds=int(config["ordinal_logistic"]["calibration_folds"]),
        lower_threshold=float(parameters["deployment_lower_threshold"]),
        upper_threshold=float(parameters["deployment_upper_threshold"]),
        version=version,
    ).fit(train_reviews, train_labels)
    predictions = model.predict(test_reviews)
    probabilities = model.predict_proba(test_reviews)
    candidate_metrics = classification_metrics(test_labels, predictions, probabilities)
    candidate_metrics["by_language"] = _metrics_by_language(
        test_languages, test_labels, predictions, probabilities
    )

    baseline_path = output_dir / "final_metrics.json"
    baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_metrics.update(_ordinal_metrics_from_confusion(baseline_metrics))
    checks = promotion_gate(candidate_metrics, baseline_metrics)
    diagnostics = ordinal_diagnostics(candidate_metrics, baseline_metrics)
    passed = all(checks.values())
    comparison = {
        "status": "existing_held_out_comparison_not_blind",
        "held_out_test_evaluated": True,
        "warning": (
            "These rows were reused by earlier research and are not a new blind benchmark."
        ),
        "selected_candidate": selected_name,
        "parameters": parameters,
        "data": {
            "raw_sha256": sha256_file(config["data_path"]),
            "training_rows": len(split.train),
            "held_out_rows": len(split.test),
            "random_seed": seed,
        },
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "promotion_gate": {"checks": checks, "passed": passed},
        "non_blocking_ordinal_diagnostics": diagnostics,
        "artifact_replaced": passed,
    }
    comparison_path = experiment_dir / "ordinal_logistic_held_out_evaluation.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    if not passed:
        return comparison

    model_path = output_dir / "model.joblib"
    old_metadata_path = output_dir / "model_metadata.json"
    old_metadata = json.loads(old_metadata_path.read_text(encoding="utf-8"))
    previous_model = {
        "model_version": old_metadata.get("model_version"),
        "model_sha256": old_metadata.get("model_sha256"),
        "experiment": old_metadata.get("experiment"),
        "held_out_metrics": baseline_metrics,
    }
    model.save(model_path)
    model_hash = sha256_file(model_path)
    model_size = model_path.stat().st_size

    misclassified = []
    for review, actual, predicted, language in zip(
        test_reviews, test_labels, predictions, test_languages, strict=True
    ):
        if actual != predicted:
            misclassified.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "detected_language": language,
                    "excerpt": " ".join(review.split())[:160].rstrip(),
                }
            )
    pd.DataFrame(misclassified).to_csv(output_dir / "error_analysis.csv", index=False)

    split_metadata = {
        **old_metadata["split"],
        "train_normalized_sha256": _hash_values(split.train["normalized_review"].tolist()),
        "test_normalized_sha256": _hash_values(split.test["normalized_review"].tolist()),
    }
    metadata = {
        **old_metadata,
        "package_version": __version__,
        "model_version": version,
        "training_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "model_sha256": model_hash,
        "model_size_bytes": model_size,
        "mlflow_run_id": None,
        "label_classes": list(LABELS),
        "expected_input_schema": {
            "review": (f"non-empty Dutch or English string <= {MAX_REVIEW_CHARACTERS} characters")
        },
        "split": split_metadata,
        "experiment": {
            "name": "ordinal_crossfit_threshold",
            "feature_kind": "combined",
            "boundaries": [
                "Negative vs Average+Positive",
                "Negative+Average vs Positive",
            ],
            **parameters,
        },
        "held_out_metrics": candidate_metrics,
        "promotion": {
            "comparison_path": str(comparison_path),
            "checks": checks,
            "passed": passed,
            "warning": comparison["warning"],
        },
        "previous_model": previous_model,
    }
    baseline_path.write_text(json.dumps(candidate_metrics, indent=2), encoding="utf-8")
    old_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ordinal_promotion(args.config)
    print(json.dumps(result["promotion_gate"], indent=2))


if __name__ == "__main__":
    main()
