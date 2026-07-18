import numpy as np
import pytest

from dutch_sentiment.ordinal_logistic_experiment import (
    _passes_blind_evaluation_gate,
    boundary_threshold_labels,
    compose_ordinal_probabilities,
    probability_argmax_labels,
    project_monotonic_boundaries,
)
from dutch_sentiment.ordinal_promote import (
    _promotion_baseline,
    ordinal_diagnostics,
    promotion_gate,
)


def test_monotonic_projection_pools_only_violations() -> None:
    lower, upper, evidence = project_monotonic_boundaries(
        np.asarray([0.8, 0.3]), np.asarray([0.2, 0.7])
    )
    assert lower.tolist() == pytest.approx([0.8, 0.5])
    assert upper.tolist() == pytest.approx([0.2, 0.5])
    assert evidence["violation_rate"] == pytest.approx(0.5)
    assert evidence["mean_violation_gap"] == pytest.approx(0.2)


def test_composed_probabilities_are_ordered_and_normalized() -> None:
    probabilities = compose_ordinal_probabilities(
        np.asarray([0.2, 0.7, 0.9]), np.asarray([0.1, 0.3, 0.8])
    )
    np.testing.assert_allclose(probabilities, [[0.8, 0.1, 0.1], [0.3, 0.4, 0.3], [0.1, 0.1, 0.8]])
    assert probability_argmax_labels(probabilities) == ["Negative", "Average", "Positive"]


def test_boundary_threshold_decisions_follow_order() -> None:
    predictions = boundary_threshold_labels(
        np.asarray([0.2, 0.8, 0.9]),
        np.asarray([0.1, 0.4, 0.8]),
        lower_threshold=0.5,
        upper_threshold=0.6,
    )
    assert predictions == ["Negative", "Average", "Positive"]


def test_boundary_thresholds_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        boundary_threshold_labels(np.asarray([0.5]), np.asarray([0.4]), 0.0, 0.5)


def test_blind_evaluation_gate_requires_all_ordinal_improvements() -> None:
    baseline = {
        "macro_f1": 0.65,
        "ordinal_mae": 0.35,
        "quadratic_weighted_kappa": 0.45,
        "severe_error_rate": 0.02,
        "per_class": {"Negative": {"precision": 0.73, "recall": 0.50, "f1-score": 0.59}},
    }
    candidate = {
        "macro_f1": 0.65,
        "ordinal_mae": 0.34,
        "quadratic_weighted_kappa": 0.47,
        "severe_error_rate": 0.021,
        "per_class": {"Negative": {"precision": 0.62, "recall": 0.60, "f1-score": 0.61}},
    }
    assert not _passes_blind_evaluation_gate(candidate, baseline)
    candidate["severe_error_rate"] = 0.019
    assert _passes_blind_evaluation_gate(candidate, baseline)


def test_production_promotion_gate_uses_predeclared_classification_constraints() -> None:
    baseline = {
        "macro_f1": 0.64,
        "balanced_accuracy": 0.62,
        "ordinal_mae": 0.36,
        "quadratic_weighted_kappa": 0.45,
        "severe_error_rate": 0.02,
        "per_class": {"Negative": {"precision": 0.72, "recall": 0.52, "f1-score": 0.60}},
    }
    candidate = {
        "macro_f1": 0.65,
        "balanced_accuracy": 0.64,
        "ordinal_mae": 0.35,
        "quadratic_weighted_kappa": 0.47,
        "severe_error_rate": 0.019,
        "per_class": {"Negative": {"precision": 0.62, "recall": 0.60, "f1-score": 0.61}},
    }
    assert all(promotion_gate(candidate, baseline).values())
    candidate["per_class"]["Negative"]["precision"] = 0.59
    checks = promotion_gate(candidate, baseline)
    assert not checks["negative_precision_at_least_0_60"]
    assert not all(checks.values())


def test_ordinal_diagnostics_are_reported_separately() -> None:
    baseline = {
        "ordinal_mae": 0.36,
        "quadratic_weighted_kappa": 0.45,
        "severe_error_rate": 0.02,
    }
    candidate = {
        "ordinal_mae": 0.361,
        "quadratic_weighted_kappa": 0.47,
        "severe_error_rate": 0.019,
    }
    assert ordinal_diagnostics(candidate, baseline) == {
        "ordinal_mae_not_higher": False,
        "quadratic_weighted_kappa_not_lower": True,
        "severe_error_rate_not_higher": True,
    }


def test_repeated_promotion_keeps_original_baseline() -> None:
    original = {"macro_f1": 0.64}
    promoted = {"macro_f1": 0.65}
    metadata = {
        "experiment": {"name": "ordinal_crossfit_threshold"},
        "previous_model": {"held_out_metrics": original},
    }
    assert _promotion_baseline(promoted, metadata) == original
    assert _promotion_baseline(original, {"experiment": {"name": "multiclass"}}) == original
