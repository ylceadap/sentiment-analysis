"""Ordinal probability composition, decision rules, and diagnostic metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

ORDERED_LABELS = ("Negative", "Average", "Positive")
ORDINAL_VALUES: dict[str, int] = {"Negative": 1, "Average": 2, "Positive": 3}


def project_monotonic_boundaries(
    above_negative: np.ndarray, above_average: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Project boundary probabilities onto P(y>Average) <= P(y>Negative)."""
    lower = np.asarray(above_negative, dtype=float).copy()
    upper = np.asarray(above_average, dtype=float).copy()
    if lower.shape != upper.shape:
        raise ValueError("Ordinal boundary probability arrays must have the same shape")
    violations = upper > lower
    original_gap = np.maximum(upper - lower, 0.0)
    # Pool each violating pair at its midpoint, the least-squares isotonic projection.
    midpoint = (lower[violations] + upper[violations]) / 2.0
    lower[violations] = midpoint
    upper[violations] = midpoint
    return (
        lower,
        upper,
        {
            "violation_rate": float(violations.mean()) if len(violations) else 0.0,
            "mean_violation_gap": float(original_gap.mean()) if len(original_gap) else 0.0,
        },
    )


def compose_ordinal_probabilities(
    above_negative: np.ndarray, above_average: np.ndarray
) -> np.ndarray:
    """Compose class probabilities from P(y>Negative) and P(y>Average)."""
    lower, upper, _ = project_monotonic_boundaries(above_negative, above_average)
    # P(N)=1-lower, P(A)=lower-upper, and P(P)=upper.
    probabilities = np.column_stack((1.0 - lower, lower - upper, upper))
    probabilities = np.clip(probabilities, 0.0, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("Ordinal probability composition produced an empty row")
    return probabilities / row_sums


def probability_argmax_labels(probabilities: np.ndarray) -> list[str]:
    """Map each probability row to the highest-probability ordered label."""
    return [ORDERED_LABELS[int(index)] for index in np.argmax(probabilities, axis=1)]


def boundary_threshold_labels(
    above_negative: np.ndarray,
    above_average: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> list[str]:
    """Apply explicit lower and upper ordinal decision thresholds."""
    if not 0 < lower_threshold < 1 or not 0 < upper_threshold < 1:
        raise ValueError("Ordinal decision thresholds must lie strictly between zero and one")
    lower, upper, _ = project_monotonic_boundaries(above_negative, above_average)
    predictions = np.where(
        lower < lower_threshold,
        "Negative",
        np.where(upper >= upper_threshold, "Positive", "Average"),
    )
    return predictions.tolist()


def with_ordinal_diagnostics(
    metrics: dict[str, Any], labels: list[str], predictions: list[str]
) -> dict[str, Any]:
    """Add ordinal MAE, quadratic kappa, and adjacent/severe error rates."""
    true_ordinal = np.asarray([ORDINAL_VALUES[label] for label in labels])
    predicted_ordinal = np.asarray([ORDINAL_VALUES[label] for label in predictions])
    # |true-predicted| is 0 for correct, 1 for adjacent, and 2 for severe errors.
    distance = np.abs(true_ordinal - predicted_ordinal)
    qwk = float(cohen_kappa_score(true_ordinal, predicted_ordinal, weights="quadratic"))
    if not np.isfinite(qwk):
        qwk = 1.0 if np.array_equal(true_ordinal, predicted_ordinal) else 0.0
    return {
        **metrics,
        "ordinal_mae": float(distance.mean()),
        "quadratic_weighted_kappa": qwk,
        "adjacent_error_rate": float(np.mean(distance == 1)),
        "severe_error_rate": float(np.mean(distance == 2)),
    }
