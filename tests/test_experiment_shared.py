from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dutch_sentiment.embedding_runtime import embedding_cache_path, encode_or_load
from dutch_sentiment.experiment_data import prepare_frozen_experiment
from dutch_sentiment.experiment_utils import (
    aligned_probabilities,
    fold_summary,
    hash_reviews,
    hash_values,
    language_slices,
    negative_metrics,
    promotion_gate,
    select_by_gate,
)


class ProbabilityEstimator:
    """Expose configurable classes and probability rows for alignment tests."""

    def __init__(self, classes: list[str], probabilities: list[list[float]]) -> None:
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, features: object) -> np.ndarray:
        """Return the configured probabilities independently of dummy features."""
        del features
        return self.probabilities


def test_hashes_and_probability_alignment_validate_inputs() -> None:
    """Shared hashes are stable and probabilities are reordered and validated."""
    assert hash_values(["a", "b"]) == hash_values(["a", "b"])
    assert hash_values(["a", "b"]) != hash_values(["b", "a"])
    assert hash_reviews(["é"]) == hash_reviews(["é"])
    estimator = ProbabilityEstimator(["Negative", "Positive", "Average"], [[0.2, 0.5, 0.3]])
    assert aligned_probabilities(estimator, None).tolist() == [[0.5, 0.3, 0.2]]
    with pytest.raises(ValueError, match="missing labels"):
        aligned_probabilities(ProbabilityEstimator(["Positive"], [[1.0]]), None)
    with pytest.raises(ValueError, match="empty row"):
        aligned_probabilities(
            ProbabilityEstimator(["Positive", "Average", "Negative"], [[0.0, 0.0, 0.0]]),
            None,
        )


def test_shared_metric_slices_and_selection_cover_guardrails() -> None:
    """Fold/language summaries and shared promotion selection preserve policy."""
    labels = ["Positive", "Average", "Negative", "Negative"]
    predictions = ["Positive", "Average", "Negative", "Average"]
    folds = [np.asarray([0, 2]), np.asarray([1, 3])]
    summary = fold_summary(labels, predictions, folds)
    assert summary["cv_macro_f1_mean"] > 0
    slices = language_slices(labels, predictions, ["dutch"] * 4)
    assert slices["dutch_rows"] == 4
    metrics = {"per_class": {"Negative": {"precision": 0.8, "recall": 0.5, "f1-score": 0.6}}}
    assert negative_metrics(metrics) == (0.8, 0.5, 0.6)

    baseline = {
        "cv_macro_f1_mean": 0.65,
        "cv_macro_f1_std": 0.02,
        "negative_recall": 0.50,
        "oof_accuracy": 0.70,
    }
    gates = {
        "minimum_macro_f1_improvement": 0.01,
        "minimum_negative_precision": 0.60,
        "minimum_negative_recall_improvement": 0.05,
        "maximum_accuracy_drop": 0.02,
        "maximum_cv_macro_f1_std_increase": 0.01,
    }
    eligible = {
        "name": "eligible",
        "cv_macro_f1_mean": 0.67,
        "cv_macro_f1_std": 0.02,
        "negative_precision": 0.65,
        "negative_recall": 0.60,
        "oof_accuracy": 0.69,
    }
    ineligible = {**eligible, "name": "ineligible", "negative_precision": 0.50}
    assert all(promotion_gate(eligible, baseline, gates).values())
    selected, checks, passed = select_by_gate(
        [ineligible, eligible],
        baseline,
        gates,
        include=lambda row: True,
        rank=lambda row: (row["cv_macro_f1_mean"], row["negative_precision"]),
    )
    assert selected["name"] == "eligible" and all(checks.values()) and passed
    with pytest.raises(ValueError, match="No experiment candidates"):
        select_by_gate(
            [eligible],
            baseline,
            gates,
            include=lambda row: False,
            rank=lambda row: (row["cv_macro_f1_mean"],),
        )


def test_embedding_cache_hit_and_hash_mismatch(tmp_path: Path) -> None:
    """Embedding cache loads verified arrays and rejects altered cache metadata."""
    reviews = ["goed", "slecht"]
    model = {"name": "demo", "model_id": "demo/id", "revision": "abc"}
    config = {
        "cache_dir": str(tmp_path),
        "normalize_embeddings": True,
        "huggingface_cache_dir": str(tmp_path / "hf"),
        "batch_size": 2,
    }
    review_hash = hash_reviews(reviews)
    variant = json.dumps(
        {"task": None, "max_sequence_length": None, "truncate_dimension": None},
        sort_keys=True,
    )
    cache = embedding_cache_path(tmp_path, "demo", "abc", review_hash, True, variant)
    np.savez_compressed(
        cache,
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        review_hash=np.asarray(review_hash),
    )
    embeddings, runtime = encode_or_load(model, reviews, config)
    assert embeddings.shape == (2, 2)
    assert runtime["cache_hit"] is True

    np.savez_compressed(
        cache,
        embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
        review_hash=np.asarray("damaged"),
    )
    with pytest.raises(RuntimeError, match="cache hash mismatch"):
        encode_or_load(model, reviews, config)


def test_prepare_frozen_experiment_reproduces_documented_split() -> None:
    """The shared preparation helper reproduces the frozen training boundary."""
    config = yaml.safe_load(Path("configs/embedding_experiment.yaml").read_text())
    prepared = prepare_frozen_experiment(config)
    assert prepared.train_rows == 3838
    assert prepared.heldout_rows == 960
    assert len(prepared.folds) == 5
    assert prepared.train_sha256 == config["expected_train_normalized_sha256"]
