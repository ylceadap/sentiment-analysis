import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbalancedPipeline

from dutch_sentiment.imbalance_experiment import (
    ImbalanceCandidate,
    build_candidate_pipeline,
    candidate_specs,
    select_candidate,
    threshold_predictions,
)

MODEL_CONFIG = {
    "min_df": 1,
    "max_df": 1.0,
    "word_max_features": 100,
    "char_max_features": 100,
    "max_iter": 100,
}


def test_threshold_predictions_promote_negative_and_exclude_it_below_threshold() -> None:
    probabilities = np.asarray(
        [
            [0.40, 0.35, 0.25],
            [0.20, 0.35, 0.45],
            [0.45, 0.40, 0.15],
        ]
    )
    assert threshold_predictions(probabilities, "argmax") == [
        "Positive",
        "Negative",
        "Positive",
    ]
    assert threshold_predictions(probabilities, 0.20) == [
        "Negative",
        "Negative",
        "Positive",
    ]
    assert threshold_predictions(probabilities, 0.50) == [
        "Positive",
        "Average",
        "Positive",
    ]
    with pytest.raises(ValueError, match="between zero and one"):
        threshold_predictions(probabilities, 1.1)


def test_candidate_specs_and_fold_local_random_oversampling() -> None:
    config = {
        "class_weight_candidates": [{"name": "balanced", "value": "balanced"}],
        "oversampling_candidates": [
            {"name": "random_oversample_negative_x2", "negative_multiplier": 2}
        ],
    }
    candidates = candidate_specs(config)
    assert [candidate.name for candidate in candidates] == [
        "balanced",
        "random_oversample_negative_x2",
    ]
    balanced = build_candidate_pipeline(candidates[0], MODEL_CONFIG, seed=42)
    assert balanced.named_steps["classifier"].class_weight == "balanced"
    labels = ["Positive", "Positive", "Average", "Average", "Negative"]
    oversampled = build_candidate_pipeline(
        candidates[1], MODEL_CONFIG, seed=42, fold_train_labels=labels
    )
    assert isinstance(oversampled, ImbalancedPipeline)
    assert oversampled.named_steps["random_oversample"].sampling_strategy == {"Negative": 2}
    assert oversampled.named_steps["classifier"].class_weight is None


def test_oversampling_rejects_missing_fold_labels() -> None:
    candidate = ImbalanceCandidate(
        "random_oversample_negative_x2", "random_oversampling", negative_multiplier=2
    )
    with pytest.raises(ValueError, match="Fold labels"):
        build_candidate_pipeline(candidate, MODEL_CONFIG, seed=42)


def test_selection_enforces_precision_then_close_recall_tiebreak() -> None:
    config = {
        "selection": {
            "minimum_negative_precision": 0.60,
            "close_negative_recall_tolerance": 0.01,
        }
    }
    rows = [
        {
            "name": "high_recall_low_precision",
            "base_candidate": "a",
            "negative_threshold": "0.20",
            "oof_negative_precision": 0.59,
            "oof_negative_recall": 0.90,
            "oof_negative_f1_score": 0.71,
            "oof_macro_f1": 0.64,
            "cv_macro_f1_mean": 0.64,
            "cv_macro_f1_std": 0.02,
            "complexity_rank": 0,
        },
        {
            "name": "best_recall",
            "base_candidate": "b",
            "negative_threshold": "0.25",
            "oof_negative_precision": 0.60,
            "oof_negative_recall": 0.80,
            "oof_negative_f1_score": 0.69,
            "oof_macro_f1": 0.63,
            "cv_macro_f1_mean": 0.63,
            "cv_macro_f1_std": 0.02,
            "complexity_rank": 1,
        },
        {
            "name": "close_better_macro_f1",
            "base_candidate": "c",
            "negative_threshold": "0.30",
            "oof_negative_precision": 0.65,
            "oof_negative_recall": 0.795,
            "oof_negative_f1_score": 0.71,
            "oof_macro_f1": 0.66,
            "cv_macro_f1_mean": 0.66,
            "cv_macro_f1_std": 0.02,
            "complexity_rank": 2,
        },
    ]
    decision = select_candidate(pd.DataFrame(rows), config)
    assert decision["eligible"] is True
    assert decision["selected_candidate"] == "close_better_macro_f1"
    assert decision["eligible_candidates"] == 2


def test_selection_handles_no_precision_eligible_candidate() -> None:
    config = {
        "selection": {
            "minimum_negative_precision": 0.60,
            "close_negative_recall_tolerance": 0.01,
        }
    }
    results = pd.DataFrame(
        [{"name": "x", "oof_negative_precision": 0.59, "oof_negative_recall": 0.9}]
    )
    decision = select_candidate(results, config)
    assert decision["eligible"] is False
    assert decision["selected_candidate"] is None
