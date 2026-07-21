from __future__ import annotations

import numpy as np

from dutch_sentiment.experiments.jina_ordinal import (
    _gate_candidate,
    _select_thresholds,
)


def test_jina_ordinal_gate_requires_all_predeclared_constraints() -> None:
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
    candidate["negative_precision"] = 0.59
    assert not _gate_candidate(candidate, baseline, gates)["negative_precision"]


def test_jina_ordinal_threshold_selection_prefers_eligible_negative_recall() -> None:
    labels = ["Negative", "Negative", "Average", "Positive", "Positive"]
    above_negative = np.asarray([0.2, 0.4, 0.8, 0.9, 0.9])
    above_average = np.asarray([0.1, 0.2, 0.3, 0.6, 0.8])
    baseline = {
        "macro_f1": 0.70,
        "per_class": {"Negative": {"recall": 0.50}},
    }
    lower, upper, metrics = _select_thresholds(
        labels,
        above_negative,
        above_average,
        baseline,
        [0.3, 0.5, 0.7],
    )
    assert lower == 0.5
    assert upper in {0.3, 0.5}
    assert metrics["per_class"]["Negative"]["precision"] >= 0.60
    assert metrics["per_class"]["Negative"]["recall"] >= 0.50
