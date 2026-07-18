import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from dutch_sentiment.linear_experiment import (
    LinearCandidate,
    build_candidate_pipeline,
    candidate_specs,
    decide_promotion,
)

MODEL_CONFIG = {
    "min_df": 1,
    "max_df": 1.0,
    "word_max_features": 100,
    "char_max_features": 100,
    "max_iter": 100,
}


def test_candidate_grid_and_classifier_construction() -> None:
    config = {
        "logistic_c_values": [0.5, 1.0],
        "linear_svc_c_values": [2.0],
    }
    candidates = candidate_specs(config)
    assert [candidate.name for candidate in candidates] == [
        "logreg_c0.5",
        "logreg_c1",
        "linear_svc_c2",
    ]
    logistic = build_candidate_pipeline(candidates[0], MODEL_CONFIG, seed=42)
    svc = build_candidate_pipeline(candidates[-1], MODEL_CONFIG, seed=42)
    assert isinstance(logistic.named_steps["classifier"], LogisticRegression)
    assert logistic.named_steps["classifier"].C == 0.5
    assert isinstance(svc.named_steps["classifier"], LinearSVC)
    assert svc.named_steps["classifier"].C == 2.0


def test_unknown_classifier_is_rejected() -> None:
    candidate = LinearCandidate("bad", "unknown", 1.0)
    with pytest.raises(ValueError, match="Unsupported classifier"):
        build_candidate_pipeline(candidate, MODEL_CONFIG, seed=42)


def test_promotion_requires_improvement_and_both_guardrails() -> None:
    config = {
        "promotion": {
            "minimum_macro_f1_improvement": 0.01,
            "maximum_dutch_macro_f1_drop": 0.01,
            "maximum_negative_f1_drop": 0.0,
        }
    }
    passing = pd.DataFrame(
        [
            {
                "name": "logreg_c1",
                "cv_macro_f1_mean": 0.64,
                "oof_dutch_macro_f1": 0.64,
                "oof_negative_f1": 0.60,
            },
            {
                "name": "linear_svc_c1",
                "cv_macro_f1_mean": 0.655,
                "oof_dutch_macro_f1": 0.635,
                "oof_negative_f1": 0.61,
            },
        ]
    )
    assert decide_promotion(passing, config)["promote"] is True

    failing = passing.copy()
    failing.loc[failing["name"].eq("linear_svc_c1"), "oof_negative_f1"] = 0.59
    decision = decide_promotion(failing, config)
    assert decision["promote"] is False
    assert decision["checks"]["negative_f1_guardrail"] is False
