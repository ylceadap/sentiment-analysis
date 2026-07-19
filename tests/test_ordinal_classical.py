from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from dutch_sentiment.models.ordinal_classical import (
    OrdinalSentimentModel,
    build_ordinal_model,
    load_sentiment_model,
)


def _training_rows() -> tuple[list[str], list[str]]:
    """Return a small balanced corpus with repeated learnable sentiment cues."""
    reviews = [
        "geweldig mooi sterk acteerwerk",
        "geweldig verhaal en mooie film",
        "sterk en prachtig einde",
        "mooie geweldige ervaring",
        "redelijk gewone film",
        "gemiddeld verhaal en tempo",
        "gewone redelijke ervaring",
        "gemiddeld maar acceptabel",
        "vreselijk slecht saai verhaal",
        "slecht acteerwerk en saai",
        "vreselijke saaie ervaring",
        "slecht en teleurstellend einde",
    ]
    labels = ["Positive"] * 4 + ["Average"] * 4 + ["Negative"] * 4
    return reviews, labels


def test_ordinal_model_fit_predict_probability_infer_and_round_trip(tmp_path: Path) -> None:
    """The archived ordinal contract remains trainable, serializable, and loadable."""
    reviews, labels = _training_rows()
    model = build_ordinal_model(
        {
            "min_df": 1,
            "max_df": 1.0,
            "word_max_features": 200,
            "char_max_features": 300,
            "max_iter": 200,
            "random_seed": 42,
        },
        c_value=1.0,
        calibration_folds=2,
        lower_threshold=0.5,
        upper_threshold=0.5,
        version="test-v1",
    ).fit(reviews, labels)

    predictions = model.predict(reviews[:3])
    probabilities = model.predict_proba(reviews[:3])
    inference = model.infer(reviews[0])

    assert len(predictions) == 3
    assert inference.label in {"Positive", "Average", "Negative"}
    assert all(sum(row.values()) == pytest.approx(1.0) for row in probabilities)
    with pytest.raises(RuntimeError, match="explanations are not supported"):
        model.infer(reviews[0], explain=True)

    artifact = tmp_path / "ordinal.joblib"
    model.save(artifact)
    loaded = load_sentiment_model(artifact)
    assert isinstance(loaded, OrdinalSentimentModel)
    assert loaded.predict(reviews[:1]) == model.predict(reviews[:1])


def test_ordinal_model_rejects_invalid_labels_and_artifacts(tmp_path: Path) -> None:
    """Invalid training labels and unrelated pickle payloads fail closed."""
    reviews, labels = _training_rows()
    model = build_ordinal_model(
        {"min_df": 1, "max_df": 1.0},
        c_value=1.0,
        calibration_folds=2,
        lower_threshold=0.5,
        upper_threshold=0.5,
        version="test-v1",
    )
    with pytest.raises(ValueError, match="Unexpected labels"):
        model.fit(reviews, [*labels[:-1], "Unknown"])
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_sentiment_model(tmp_path / "missing.joblib")

    unrelated = tmp_path / "unrelated.joblib"
    joblib.dump({"not": "a model"}, unrelated)
    with pytest.raises(TypeError, match="not a supported sentiment model"):
        load_sentiment_model(unrelated)


def test_boundary_probabilities_project_order_violations() -> None:
    """Crossed cumulative boundaries are projected to their shared midpoint."""

    class Boundary:
        def __init__(self, positive: list[float]) -> None:
            self.positive = np.asarray(positive)

        def predict_proba(self, transformed: object) -> np.ndarray:
            return np.column_stack([1 - self.positive, self.positive])

    lower, upper = OrdinalSentimentModel._boundary_probabilities(
        Boundary([0.2, 0.8]), Boundary([0.6, 0.3]), object()
    )

    assert lower.tolist() == pytest.approx([0.4, 0.8])
    assert upper.tolist() == pytest.approx([0.4, 0.3])
