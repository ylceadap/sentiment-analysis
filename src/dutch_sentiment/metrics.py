"""Consistent classification metrics for experiments and held-out evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
)

from .constants import LABELS

ORDINAL_VALUES: dict[str, int] = {"Negative": 1, "Average": 2, "Positive": 3}

CV_SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_precision": make_scorer(precision_score, average="macro", zero_division=0),
    "macro_recall": make_scorer(recall_score, average="macro", zero_division=0),
    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    "weighted_f1": make_scorer(f1_score, average="weighted", zero_division=0),
}


def _probability_metrics(
    y_true: list[str], probabilities: list[dict[str, float]], bins: int = 10
) -> dict[str, float]:
    matrix = np.asarray([[row[label] for label in LABELS] for row in probabilities])
    target_indices = np.asarray([LABELS.index(label) for label in y_true])
    one_hot = np.eye(len(LABELS))[target_indices]
    confidence = matrix.max(axis=1)
    predicted_indices = matrix.argmax(axis=1)
    correct = predicted_indices == target_indices
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    calibration_error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            calibration_error += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )
    return {
        "log_loss": float(log_loss(target_indices, matrix, labels=list(range(len(LABELS))))),
        "multiclass_brier_score": float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1))),
        "expected_calibration_error_10_bin": calibration_error,
        "mean_prediction_confidence": float(confidence.mean()),
    }


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    probabilities: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Return aggregate, per-class, and confusion-matrix evidence."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    true_ordinal = np.asarray([ORDINAL_VALUES[label] for label in y_true])
    predicted_ordinal = np.asarray([ORDINAL_VALUES[label] for label in y_pred])
    ordinal_distance = np.abs(true_ordinal - predicted_ordinal)
    qwk = float(cohen_kappa_score(true_ordinal, predicted_ordinal, weights="quadratic"))
    if not np.isfinite(qwk):
        qwk = 1.0 if np.array_equal(true_ordinal, predicted_ordinal) else 0.0
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        "label_order": list(LABELS),
        "ordinal_mae": float(ordinal_distance.mean()),
        "quadratic_weighted_kappa": qwk,
        "adjacent_error_rate": float(np.mean(ordinal_distance == 1)),
        "severe_error_rate": float(np.mean(ordinal_distance == 2)),
    }
    if probabilities is not None:
        metrics.update(_probability_metrics(y_true, probabilities))
    return metrics
