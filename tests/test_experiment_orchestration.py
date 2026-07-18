from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from dutch_sentiment import embedding_experiment, jina_ordinal_logistic_experiment, train


def _prepared() -> SimpleNamespace:
    """Return a minimal hash-verified experiment fixture."""
    return SimpleNamespace(
        reviews=["goed", "matig", "slecht"],
        labels=["Positive", "Average", "Negative"],
        languages=["dutch", "dutch", "dutch"],
        folds=[(np.asarray([0, 1]), np.asarray([2]))],
        training_config={"model": {}},
        seed=42,
        raw_sha256="raw",
        train_sha256="train",
        heldout_sha256="heldout",
        train_rows=3,
        heldout_rows=1,
    )


def _baseline() -> dict[str, object]:
    """Return the common baseline fields required by both selection workflows."""
    return {
        "name": "baseline",
        "model_type": "tfidf_word_char_logreg",
        "model_id": "baseline",
        "cv_macro_f1_mean": 0.60,
        "cv_macro_f1_std": 0.02,
        "oof_accuracy": 0.60,
        "negative_precision": 0.70,
        "negative_recall": 0.40,
        "negative_f1": 0.50,
    }


def test_embedding_run_writes_portable_decision_without_holdout(
    monkeypatch, tmp_path: Path
) -> None:
    """Embedding orchestration writes evidence while preserving the holdout boundary."""
    config = {
        "training_config": "unused",
        "random_seed": 42,
        "models": [{"name": "demo", "model_id": "demo/id", "revision": "r1", "license": "x"}],
        "promotion_gates": {
            "minimum_macro_f1_improvement": 0.01,
            "minimum_negative_precision": 0.60,
            "minimum_negative_recall_improvement": 0.05,
            "maximum_accuracy_drop": 0.02,
            "maximum_cv_macro_f1_std_increase": 0.01,
        },
        "output_csv": str(tmp_path / "results.csv"),
        "decision_json": str(tmp_path / "decision.json"),
        "report_path": str(tmp_path / "report.md"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    candidate = {
        **_baseline(),
        "name": "demo-candidate",
        "model_type": "frozen_sentence_embedding_logreg",
        "model_id": "demo/id",
        "revision": "r1",
        "cv_macro_f1_mean": 0.70,
        "negative_precision": 0.70,
        "negative_recall": 0.60,
        "oof_accuracy": 0.65,
        "dutch_macro_f1": 0.70,
        "english_macro_f1": 0.50,
    }
    monkeypatch.setattr(embedding_experiment, "prepare_frozen_experiment", lambda _: _prepared())
    monkeypatch.setattr(embedding_experiment, "_official_baseline", lambda *args: _baseline())
    monkeypatch.setattr(
        embedding_experiment,
        "_encode_or_load",
        lambda *args: (np.ones((3, 2)), {"cache_hit": True}),
    )
    monkeypatch.setattr(embedding_experiment, "_embedding_rows", lambda *args: [candidate])
    monkeypatch.setattr(
        embedding_experiment,
        "_write_report",
        lambda *args: Path(args[-1]).write_text("report"),
    )
    decision = embedding_experiment.run_experiment(path)
    assert decision["metric_gates_passed"] is True
    assert decision["heldout_evaluated"] is False
    assert json.loads((tmp_path / "decision.json").read_text())["selected_candidate"]
    assert (tmp_path / "results.csv").is_file()


def test_jina_ordinal_run_writes_research_only_decision(monkeypatch, tmp_path: Path) -> None:
    """Jina ordinal orchestration records provenance without production promotion."""
    config = {
        "training_config": "unused",
        "random_seed": 42,
        "models": [{"name": "jina", "model_id": "jina/id", "revision": "r1", "license": "nc"}],
        "promotion_gates": {
            "minimum_macro_f1_improvement": 0.01,
            "minimum_negative_precision": 0.60,
            "minimum_negative_recall_improvement": 0.05,
            "maximum_accuracy_drop": 0.02,
            "maximum_cv_macro_f1_std_increase": 0.01,
        },
        "output_csv": str(tmp_path / "ordinal.csv"),
        "decision_json": str(tmp_path / "ordinal.json"),
        "report_path": str(tmp_path / "ordinal.md"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    common = {
        **_baseline(),
        "model_id": "jina/id",
        "revision": "r1",
        "cv_macro_f1_mean": 0.70,
        "negative_precision": 0.70,
        "negative_recall": 0.60,
        "negative_f1": 0.64,
        "oof_accuracy": 0.65,
        "quadratic_weighted_kappa": 0.50,
        "severe_error_rate": 0.01,
        "ordinal_mae": 0.30,
        "dutch_macro_f1": 0.70,
        "english_macro_f1": 0.50,
    }
    multiclass = {
        **common,
        "name": "multiclass",
        "model_type": "jina_embedding_multiclass_logistic",
        "C": 1.0,
    }
    ordinal = {
        **common,
        "name": "ordinal",
        "model_type": "jina_embedding_two_boundary_ordinal_logistic",
        "C": 1.0,
        "cv_macro_f1_mean": 0.72,
    }
    monkeypatch.setattr(
        jina_ordinal_logistic_experiment, "prepare_frozen_experiment", lambda _: _prepared()
    )
    monkeypatch.setattr(
        jina_ordinal_logistic_experiment, "_official_baseline", lambda *args: _baseline()
    )
    monkeypatch.setattr(
        jina_ordinal_logistic_experiment,
        "_encode_or_load",
        lambda *args: (np.ones((3, 2)), {"cache_hit": True}),
    )
    monkeypatch.setattr(
        jina_ordinal_logistic_experiment,
        "_embedding_multiclass_rows",
        lambda *args: [multiclass],
    )
    monkeypatch.setattr(
        jina_ordinal_logistic_experiment,
        "_oof_multiclass_probabilities",
        lambda *args: np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
    )
    monkeypatch.setattr(
        jina_ordinal_logistic_experiment,
        "_embedding_ordinal_rows",
        lambda *args: [ordinal],
    )
    monkeypatch.setattr(
        jina_ordinal_logistic_experiment,
        "_write_report",
        lambda *args: Path(args[-1]).write_text("report"),
    )
    decision = jina_ordinal_logistic_experiment.run_experiment(path)
    assert decision["selected_candidate"] == "ordinal"
    assert decision["production_promotion_eligible"] is False
    assert decision["heldout_evaluated"] is False


def test_training_orchestration_persists_final_evidence(monkeypatch, tmp_path: Path) -> None:
    """Training orchestration selects, evaluates, and writes portable evidence under mocks."""
    labels = ["Positive", "Average", "Negative"]
    train_rows = [
        {
            "Reviews": f"{label}-{index}",
            "Label": label,
            "detected_language": language,
            "normalized_review": f"{label}-{index}",
        }
        for language in ("dutch", "english")
        for label in labels
        for index in range(2)
    ]
    test_rows = [
        {
            "Reviews": f"{label}-test-{language}",
            "Label": label,
            "detected_language": language,
            "normalized_review": f"{label}-test-{language}",
        }
        for language in ("dutch", "english")
        for label in labels
    ]
    train_frame = pd.DataFrame(train_rows)
    test_frame = pd.DataFrame(test_rows)
    raw = pd.concat([train_frame, test_frame], ignore_index=True)[["Reviews", "Label"]]
    annotated = pd.concat([train_frame, test_frame], ignore_index=True)
    split = SimpleNamespace(
        train=train_frame,
        test=test_frame,
        duplicate_rows_removed=0,
        conflicting_groups_removed=0,
    )
    config = {
        "random_seed": 42,
        "output_dir": str(tmp_path / "artifacts"),
        "report_dir": str(tmp_path / "reports"),
        "data_path": str(tmp_path / "data.csv"),
        "language": {
            "minimum_dutch_confidence": 0.7,
            "minimum_margin": 0.2,
            "short_text_characters": 20,
        },
        "test_size": 0.2,
        "model": {},
        "cv_folds": 2,
        "mlflow_tracking_uri": "sqlite:///unused.db",
        "experiment_name": "test",
    }

    class FakeRun:
        """Provide the MLflow run ID used by final metadata."""

        def __enter__(self):
            self.info = SimpleNamespace(run_id="final-run")
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeClassifier:
        """Expose fitted label classes for metadata generation."""

        classes_ = np.asarray(labels)

    class FakeModel:
        """Implement the SentimentModel methods used by orchestration."""

        def __init__(self, pipeline: object, version: str) -> None:
            del pipeline
            self.version = version
            self.pipeline = SimpleNamespace(named_steps={"classifier": FakeClassifier()})

        def fit(self, reviews: list[str], target: list[str]):
            del reviews, target
            return self

        def predict(self, reviews: list[str]) -> list[str]:
            return [review.split("-", 1)[0] for review in reviews]

        def predict_proba(self, reviews: list[str]) -> list[dict[str, float]]:
            return [
                {label: 0.9 if review.startswith(label) else 0.05 for label in labels}
                for review in reviews
            ]

        def save(self, path: Path) -> None:
            path.write_bytes(b"model")

    monkeypatch.setattr(train, "load_config", lambda _: config)
    monkeypatch.setattr(train, "load_dataset", lambda _: raw)
    monkeypatch.setattr(train, "annotate_review_languages", lambda *args: (annotated, annotated))
    monkeypatch.setattr(train, "make_holdout_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(train, "build_pipeline", lambda *args: object())
    monkeypatch.setattr(train, "SentimentModel", FakeModel)
    monkeypatch.setattr(train, "sha256_file", lambda _: "sha256")
    monkeypatch.setattr(train, "_git_commit", lambda: "1234567890abcdef")
    monkeypatch.setattr(train, "_git_dirty", lambda: False)
    monkeypatch.setattr(
        train,
        "cross_validate",
        lambda *args, **kwargs: {
            f"test_{metric}": np.asarray([0.6, 0.7]) for metric in train.CV_SCORING
        },
    )
    monkeypatch.setattr(train, "_log_candidate", lambda *args: ("candidate-run", 100))
    monkeypatch.setattr(train.mlflow, "set_tracking_uri", lambda *args: None)
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda *args: None)
    monkeypatch.setattr(train.mlflow, "start_run", lambda *args, **kwargs: FakeRun())
    monkeypatch.setattr(train.mlflow, "log_params", lambda *args: None)
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda *args: None)
    monkeypatch.setattr(train.mlflow, "log_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "build_model_report", lambda *args: None)

    metadata = train.run_training("unused.yaml")
    output = tmp_path / "artifacts"
    assert metadata["mlflow_run_id"] == "final-run"
    assert metadata["held_out_metrics"]["accuracy"] == 1.0
    assert (output / "model.joblib").read_bytes() == b"model"
    assert (output / "model_metadata.json").is_file()
