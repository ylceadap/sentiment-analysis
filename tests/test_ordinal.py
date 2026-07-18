from __future__ import annotations

import numpy as np
import pytest

from dutch_sentiment.ordinal import (
    boundary_threshold_labels,
    compose_ordinal_probabilities,
    probability_argmax_labels,
    project_monotonic_boundaries,
    with_ordinal_diagnostics,
)


def test_monotonic_projection_and_probability_equations() -> None:
    """Violating boundaries are pooled and compose normalized class probabilities."""
    lower, upper, evidence = project_monotonic_boundaries(
        np.asarray([0.2, 0.8]), np.asarray([0.6, 0.3])
    )
    assert lower.tolist() == [0.4, 0.8]
    assert upper.tolist() == [0.4, 0.3]
    assert evidence["violation_rate"] == 0.5
    probabilities = compose_ordinal_probabilities(lower, upper)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probability_argmax_labels(probabilities) == ["Negative", "Average"]
    with pytest.raises(ValueError, match="same shape"):
        project_monotonic_boundaries(np.asarray([0.1]), np.asarray([0.1, 0.2]))


def test_threshold_boundaries_and_ordinal_diagnostics() -> None:
    """Threshold equality and ordinal error distances follow documented semantics."""
    lower = np.asarray([0.5, 0.8, 0.8])
    upper = np.asarray([0.1, 0.5, 0.7])
    assert boundary_threshold_labels(lower, upper, 0.5, 0.5) == [
        "Average",
        "Positive",
        "Positive",
    ]
    with pytest.raises(ValueError, match="strictly between"):
        boundary_threshold_labels(lower, upper, 0.0, 0.5)
    diagnostics = with_ordinal_diagnostics(
        {}, ["Negative", "Average", "Positive"], ["Positive", "Average", "Average"]
    )
    assert diagnostics["ordinal_mae"] == 1.0
    assert diagnostics["severe_error_rate"] == pytest.approx(1 / 3)
