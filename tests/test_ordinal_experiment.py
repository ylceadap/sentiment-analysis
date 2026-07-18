import numpy as np
import pytest

from dutch_sentiment.ordinal_experiment import cost_sensitive_labels, scores_to_labels


def test_scores_to_labels_uses_ordered_thresholds() -> None:
    assert scores_to_labels(np.asarray([0.8, 1.49, 1.5, 2.49, 2.5, 3.2]), 1.5, 2.5) == [
        "Negative",
        "Negative",
        "Average",
        "Average",
        "Positive",
        "Positive",
    ]


def test_scores_to_labels_rejects_reversed_thresholds() -> None:
    with pytest.raises(ValueError, match="lower ordinal threshold"):
        scores_to_labels(np.asarray([2.0]), 2.5, 1.5)


def test_cost_sensitive_decision_respects_severe_error_cost() -> None:
    probabilities = np.asarray([[0.48, 0.04, 0.48]])
    labels = ["Negative", "Average", "Positive"]
    assert cost_sensitive_labels(probabilities, labels, severe_cost=1) == ["Negative"]
    assert cost_sensitive_labels(probabilities, labels, severe_cost=4) == ["Average"]
