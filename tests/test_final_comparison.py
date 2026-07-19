from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dutch_sentiment import final_comparison


class _FakeModel:
    """Return deterministic labels and probabilities for orchestration tests."""

    pipeline = SimpleNamespace(steps=())

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels

    def predict(self, reviews: list[str]) -> list[str]:
        return self.labels[: len(reviews)]

    def predict_proba(self, reviews: list[str]) -> list[dict[str, float]]:
        return [
            {label: float(label == prediction) for label in final_comparison.LABELS}
            for prediction in self.labels[: len(reviews)]
        ]


class _FakeBoundary:
    """Provide fixed binary boundary probabilities without calibration work."""

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability

    def fit(self, features: np.ndarray, labels: np.ndarray) -> _FakeBoundary:
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(len(features), self.positive_probability)
        return np.column_stack([1 - positive, positive])


def test_restore_loaded_logistic_state_repairs_legacy_multiclass_head() -> None:
    """A loaded multiclass head remains usable when sklearn omitted retired state."""
    classifier = LogisticRegression()
    classifier.classes_ = pd.Series(["Average", "Negative", "Positive"]).to_numpy()

    restored = final_comparison._restore_loaded_logistic_state(classifier)

    assert restored is classifier
    assert classifier.multi_class == "multinomial"


def test_deepseek_predictions_require_source_row_hash_and_actual_label(monkeypatch) -> None:
    """Archived API labels are accepted only after independent row-identity checks."""
    review = "  Goede\nfilm  "
    normalized = "Goede film"
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    csv = (
        "split,source_row,status,review_sha256,actual,label\n"
        f"heldout,9,ok,{digest},Positive,Average\n"
    ).encode()
    monkeypatch.setattr(final_comparison, "_git_bytes", lambda *args: csv)
    heldout = pd.DataFrame([{"source_row": 9, "Reviews": review, "Label": "Positive"}])

    assert final_comparison._deepseek_predictions(
        heldout,
        {"deepseek_archive": "archive", "deepseek_predictions": "predictions.csv"},
    ) == ["Average"]


def test_common_metrics_include_ordinal_and_language_evidence() -> None:
    """Every final candidate receives the same classification and ordinal metrics."""
    metrics = final_comparison._metrics(
        ["Positive", "Average", "Negative"] * 2,
        ["Positive", "Negative", "Negative"] * 2,
        ["dutch"] * 3 + ["english"] * 3,
    )

    assert metrics["macro_f1"] > 0
    assert metrics["ordinal_mae"] == 1 / 3
    assert set(metrics["by_language"]) == {"dutch", "english"}


def test_report_discloses_reused_holdout(tmp_path: Path) -> None:
    """The presentation report cannot imply that reused evidence is a new blind test."""
    summary = pd.DataFrame(
        [{"rank": 1, "model": "demo", "macro_f1": 0.5, "log_loss": float("nan")}]
    )
    path = tmp_path / "report.md"
    final_comparison._write_report(summary, {"data": {"heldout_rows": 3}}, path)

    content = path.read_text()
    assert "reused-heldout presentation comparison" in content
    assert "not a new blind test" in content
    assert "post_evaluation_tuning_allowed=false" in content
    assert " nan " not in content
    assert "—" in content


def test_log_existing_comparison_validates_and_logs_downloaded_result(
    tmp_path: Path, monkeypatch
) -> None:
    """Downloaded GPU evidence can be logged locally without running Jina again."""
    output_dir = tmp_path / "final"
    output_dir.mkdir()
    heldout_hash = "frozen-heldout"
    result = {
        "evaluation_scope": "reused-heldout-presentation-comparison",
        "selected_models": list(final_comparison.MODEL_ORDER),
        "data": {"heldout_normalized_sha256": heldout_hash},
    }
    (output_dir / "comparison.json").write_text(json.dumps(result))
    config = {
        "output_dir": str(output_dir),
        "expected_test_normalized_sha256": heldout_hash,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join(f"{key}: {value}" for key, value in config.items()))
    monkeypatch.setattr(final_comparison, "_log_mlflow", lambda *args: "run-123")

    logged = final_comparison.log_existing_comparison(config_path)

    assert logged["mlflow_run_id"] == "run-123"
    assert json.loads((output_dir / "comparison.json").read_text())["mlflow_run_id"] == "run-123"


def test_jina_predictions_persist_frozen_heads_without_real_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    """Jina orchestration aligns both heads and writes a hash-bound artifact."""
    labels = ["Positive", "Average", "Negative"] * 3
    train = pd.DataFrame({"Reviews": [f"train {i}" for i in range(9)], "Label": labels})
    heldout = pd.DataFrame(
        {"Reviews": ["heldout a", "heldout b", "heldout c"], "Label": labels[:3]}
    )
    embeddings = np.arange(24, dtype=np.float32).reshape(12, 2)
    monkeypatch.setattr(
        final_comparison,
        "encode_or_load",
        lambda *args: (embeddings, {"encoding_device": "cuda"}),
    )
    boundaries = iter([_FakeBoundary(0.75), _FakeBoundary(0.25)])
    monkeypatch.setattr(
        final_comparison, "_calibrated_boundary_classifier", lambda **kwargs: next(boundaries)
    )
    config = {
        "random_seed": 42,
        "jina": {
            "model": {"name": "frozen"},
            "logistic": {"C": 1.0, "class_weight": "balanced"},
            "ordinal": {"C": 1.0, "calibration_folds": 2},
        },
    }

    predictions, runtime = final_comparison._jina_predictions(train, heldout, config, tmp_path)

    assert set(predictions) == {"Jina Logistic", "Jina Ordinal"}
    assert all(len(values[0]) == 3 for values in predictions.values())
    assert Path(runtime["heads_artifact"]).is_file()
    assert len(runtime["heads_sha256"]) == 64
    assert runtime["scikit_learn"]


def test_run_comparison_writes_ranked_evidence_and_logs_mlflow(tmp_path: Path, monkeypatch) -> None:
    """The top-level comparison writes all durable outputs from frozen predictions."""
    labels = ["Positive", "Average", "Negative"] * 2
    train = pd.DataFrame({"Reviews": ["train"], "Label": ["Positive"]})
    heldout = pd.DataFrame(
        {
            "source_row": range(6),
            "Reviews": [f"review {index}" for index in range(6)],
            "Label": labels,
            "detected_language": ["dutch"] * 3 + ["english"] * 3,
        }
    )
    provenance = {
        "raw_sha256": "raw",
        "train_normalized_sha256": "train",
        "heldout_normalized_sha256": "heldout",
        "train_rows": 1,
        "heldout_rows": 6,
        "random_seed": 42,
    }
    model = _FakeModel(labels)
    monkeypatch.setattr(
        final_comparison, "_prepare_split", lambda config: (train, heldout, provenance)
    )
    monkeypatch.setattr(final_comparison.SentimentModel, "load", lambda path: model)
    monkeypatch.setattr(final_comparison, "_load_archived_model", lambda *args: model)
    monkeypatch.setattr(
        final_comparison,
        "_jina_predictions",
        lambda *args: (
            {
                "Jina Logistic": (labels, model.predict_proba(labels)),
                "Jina Ordinal": (labels, model.predict_proba(labels)),
            },
            {"encoding_device": "cuda"},
        ),
    )
    monkeypatch.setattr(final_comparison, "_deepseek_predictions", lambda *args: labels)
    monkeypatch.setattr(final_comparison, "_log_mlflow", lambda *args: "final-run")
    output_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.md"
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {output_dir}",
                f"report_path: {report_path}",
                "production_artifact: production.joblib",
                "ordinal_archive: archive",
                "ordinal_artifact: ordinal.joblib",
                "deepseek_archive: deepseek",
                "deepseek_predictions: predictions.csv",
            ]
        )
    )

    result = final_comparison.run_comparison(config_path, log_mlflow=True)

    assert result["mlflow_run_id"] == "final-run"
    assert len(result["ranking"]) == 5
    assert (output_dir / "comparison.csv").is_file()
    assert (output_dir / "heldout_predictions.csv").is_file()
    assert report_path.is_file()
