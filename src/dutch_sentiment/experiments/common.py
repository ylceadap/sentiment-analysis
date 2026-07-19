"""Shared hashing, probability, metric, and selection helpers for experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import numpy as np

from ..constants import LABELS
from ..metrics import classification_metrics


def hash_values(values: list[str]) -> str:
    """Hash an ordered string sequence using an unambiguous newline join."""
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def hash_reviews(values: list[str]) -> str:
    """Hash reviews using stable compact JSON so Unicode content is preserved."""
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def aligned_probabilities(estimator: Any, features: Any) -> np.ndarray:
    """Return estimator probabilities in the repository's canonical label order."""
    raw = estimator.predict_proba(features)
    columns = {str(label): raw[:, index] for index, label in enumerate(estimator.classes_)}
    missing = [label for label in LABELS if label not in columns]
    if missing:
        raise ValueError(f"Estimator probabilities are missing labels: {missing}")
    aligned = np.column_stack([columns[label] for label in LABELS])
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Estimator probabilities contain an empty row")
    return aligned / row_sums


def fold_summary(
    labels: list[str], predictions: list[str], validation_folds: list[np.ndarray]
) -> dict[str, float]:
    """Summarize macro-F1, balanced accuracy, and accuracy across fixed folds."""
    metrics = [
        classification_metrics(
            [labels[index] for index in fold],
            [predictions[index] for index in fold],
        )
        for fold in validation_folds
    ]
    return {
        "cv_macro_f1_mean": float(np.mean([item["macro_f1"] for item in metrics])),
        "cv_macro_f1_std": float(np.std([item["macro_f1"] for item in metrics])),
        "cv_balanced_accuracy_mean": float(
            np.mean([item["balanced_accuracy"] for item in metrics])
        ),
        "cv_balanced_accuracy_std": float(np.std([item["balanced_accuracy"] for item in metrics])),
        "cv_accuracy_mean": float(np.mean([item["accuracy"] for item in metrics])),
        "cv_accuracy_std": float(np.std([item["accuracy"] for item in metrics])),
    }


def language_slices(
    labels: list[str], predictions: list[str], languages: list[str]
) -> dict[str, float]:
    """Calculate descriptive metrics for every detected-language slice."""
    result: dict[str, float] = {}
    for language in sorted(set(languages)):
        indices = [index for index, value in enumerate(languages) if value == language]
        metrics = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
        result[f"{language}_rows"] = float(len(indices))
        result[f"{language}_macro_f1"] = metrics["macro_f1"]
        result[f"{language}_accuracy"] = metrics["accuracy"]
        result[f"{language}_negative_recall"] = metrics["per_class"]["Negative"]["recall"]
    return result


def negative_metrics(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """Extract Negative precision, recall, and F1 from a metrics document."""
    negative = metrics["per_class"]["Negative"]
    return (
        float(negative["precision"]),
        float(negative["recall"]),
        float(negative["f1-score"]),
    )


def promotion_gate(
    candidate: dict[str, Any], baseline: dict[str, Any], gates: dict[str, float]
) -> dict[str, bool]:
    """Apply the shared accuracy, minority-class, improvement, and stability gates."""
    return {
        "macro_f1": candidate["cv_macro_f1_mean"]
        >= baseline["cv_macro_f1_mean"] + gates["minimum_macro_f1_improvement"],
        "negative_precision": candidate["negative_precision"]
        >= gates["minimum_negative_precision"],
        "negative_recall": candidate["negative_recall"]
        >= baseline["negative_recall"] + gates["minimum_negative_recall_improvement"],
        "accuracy": candidate["oof_accuracy"]
        >= baseline["oof_accuracy"] - gates["maximum_accuracy_drop"],
        "stability": candidate["cv_macro_f1_std"]
        <= baseline["cv_macro_f1_std"] + gates["maximum_cv_macro_f1_std_increase"],
    }


def select_by_gate(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    gates: dict[str, float],
    *,
    include: Callable[[dict[str, Any]], bool],
    rank: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> tuple[dict[str, Any], dict[str, bool], bool]:
    """Select the highest-ranked eligible row, falling back to the best ineligible row."""
    evaluated = [(row, promotion_gate(row, baseline, gates)) for row in rows if include(row)]
    if not evaluated:
        raise ValueError("No experiment candidates matched the selection scope")
    eligible = [item for item in evaluated if all(item[1].values())]
    selected, checks = max(eligible or evaluated, key=lambda item: rank(item[0]))
    return selected, checks, bool(eligible)
