from __future__ import annotations

import numpy as np

from dutch_sentiment.embedding_experiment import _gate_candidate, threshold_predictions
from dutch_sentiment.embedding_runtime import embedding_cache_path


def test_negative_threshold_overrides_argmax() -> None:
    probabilities = np.asarray([[0.60, 0.25, 0.15], [0.45, 0.35, 0.20]])
    assert threshold_predictions(probabilities, "argmax") == ["Positive", "Positive"]
    assert threshold_predictions(probabilities, 0.20) == ["Positive", "Negative"]


def test_cache_key_changes_with_revision_or_data(tmp_path) -> None:
    original = embedding_cache_path(tmp_path, "model", "rev-1", "data-1", True)
    assert original != embedding_cache_path(tmp_path, "model", "rev-2", "data-1", True)
    assert original != embedding_cache_path(tmp_path, "model", "rev-1", "data-2", True)
    assert original != embedding_cache_path(tmp_path, "model", "rev-1", "data-1", True, "task=v2")


def test_promotion_gate_requires_every_guardrail() -> None:
    baseline = {
        "cv_macro_f1_mean": 0.65,
        "cv_macro_f1_std": 0.02,
        "negative_recall": 0.50,
        "oof_accuracy": 0.70,
    }
    candidate = {
        "cv_macro_f1_mean": 0.67,
        "cv_macro_f1_std": 0.025,
        "negative_precision": 0.61,
        "negative_recall": 0.59,
        "oof_accuracy": 0.695,
    }
    gates = {
        "minimum_macro_f1_improvement": 0.015,
        "minimum_negative_precision": 0.60,
        "minimum_negative_recall_improvement": 0.08,
        "maximum_accuracy_drop": 0.01,
        "maximum_cv_macro_f1_std_increase": 0.01,
    }
    assert all(_gate_candidate(candidate, baseline, gates).values())
