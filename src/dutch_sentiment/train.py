"""Run leakage-safe tracked experiments and train the selected final model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import __version__
from .config import load_config
from .constants import MAX_REVIEW_CHARACTERS
from .data import annotate_review_languages, load_dataset, make_holdout_split, sha256_file
from .language import DutchLanguageDetector
from .metrics import CV_SCORING, classification_metrics
from .model import ModelSpec, SentimentModel, build_pipeline
from .reporting import build_model_report

LOGGER = logging.getLogger(__name__)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _hash_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _experiment_specs() -> list[ModelSpec]:
    return [
        ModelSpec("dummy_prior", "word", dummy=True),
        ModelSpec("word_logreg", "word"),
        ModelSpec("char_logreg", "char"),
        ModelSpec("combined_logreg", "combined"),
        ModelSpec("combined_balanced_ratings", "combined", "balanced", False),
        ModelSpec("combined_balanced_masked_ratings", "combined", "balanced", True),
    ]


def _metrics_by_language(
    languages: list[str],
    labels: list[str],
    predictions: list[str],
    probabilities: list[dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """Return descriptive held-out slices for each detected language."""
    slices: dict[str, dict[str, Any]] = {}
    for language in sorted(set(languages)):
        indices = [index for index, value in enumerate(languages) if value == language]
        slices[language] = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
            [probabilities[index] for index in indices],
        )
    return slices


def _log_candidate(
    spec: ModelSpec,
    pipeline: Any,
    metrics: dict[str, float],
    params: dict[str, Any],
    train_reviews: list[str],
    train_labels: list[str],
) -> tuple[str, int]:
    with mlflow.start_run(run_name=spec.name) as run:
        mlflow.log_params({**params, **spec.__dict__})
        mlflow.log_metrics(metrics)
        pipeline.fit(train_reviews, train_labels)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "candidate.joblib"
            joblib.dump(pipeline, path, compress=3)
            size = path.stat().st_size
            mlflow.log_metric("artifact_size_bytes", size)
            mlflow.log_artifact(str(path), artifact_path="model")
        return run.info.run_id, size


def run_training(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["random_seed"])
    output_dir = Path(config["output_dir"])
    report_dir = Path(config["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(config["data_path"])
    detector = DutchLanguageDetector(**config["language"])
    annotated, language_evidence = annotate_review_languages(raw, detector)
    detected_languages = set(annotated["detected_language"].dropna())
    unsupported_languages = sorted(detected_languages - {"dutch", "english"})
    if unsupported_languages:
        raise ValueError(
            f"Training data contains unsupported detected languages: {unsupported_languages}"
        )
    split = make_holdout_split(
        annotated,
        test_size=float(config["test_size"]),
        random_seed=seed,
        stratify_columns=("detected_language", "Label"),
    )
    train_reviews = split.train["Reviews"].astype(str).tolist()
    train_labels = split.train["Label"].astype(str).tolist()
    test_reviews = split.test["Reviews"].astype(str).tolist()
    test_labels = split.test["Label"].astype(str).tolist()
    train_strata = split.train[["detected_language", "Label"]].astype(str).agg("::".join, axis=1)
    test_languages = split.test["detected_language"].fillna("unknown").astype(str).tolist()
    model_config = {**config["model"], "random_seed": seed}
    cv = StratifiedKFold(n_splits=int(config["cv_folds"]), shuffle=True, random_state=seed)
    cv_splits = list(cv.split(train_reviews, train_strata))

    tracking_uri = str(config["mlflow_tracking_uri"])
    if "://" in tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        tracking_path = Path(tracking_uri).resolve()
        tracking_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_path.as_uri())
    mlflow.set_experiment(config["experiment_name"])
    raw_hash = sha256_file(config["data_path"])
    git_commit = _git_commit()
    git_dirty = _git_dirty()
    shared_params = {
        "raw_data_sha256": raw_hash,
        "random_seed": seed,
        "cv_folds": int(config["cv_folds"]),
        "training_rows": len(split.train),
        "input_rows_after_language_annotation": len(annotated),
        "detected_dutch_rows": int(annotated["detected_language"].eq("dutch").sum()),
        "detected_english_rows": int(annotated["detected_language"].eq("english").sum()),
        "git_commit": git_commit or "unavailable",
        "git_dirty": git_dirty if git_dirty is not None else "unavailable",
    }
    rows: list[dict[str, Any]] = []
    for spec in _experiment_specs():
        LOGGER.info("Cross-validating %s", spec.name)
        pipeline = build_pipeline(spec, model_config)
        scores = cross_validate(
            pipeline,
            train_reviews,
            train_labels,
            cv=cv_splits,
            scoring=CV_SCORING,
            n_jobs=1,
            error_score="raise",
        )
        summary: dict[str, float] = {}
        row: dict[str, Any] = {
            "name": spec.name,
            "feature_kind": spec.feature_kind,
            "class_weight": spec.class_weight or "none",
            "mask_ratings": spec.mask_ratings,
        }
        for metric_name in CV_SCORING:
            values = scores[f"test_{metric_name}"]
            mean = float(values.mean())
            std = float(values.std())
            summary[f"cv_{metric_name}_mean"] = mean
            summary[f"cv_{metric_name}_std"] = std
            row[f"cv_{metric_name}_mean"] = mean
            row[f"cv_{metric_name}_std"] = std
        run_id, artifact_size = _log_candidate(
            spec, pipeline, summary, shared_params, train_reviews, train_labels
        )
        row["mlflow_run_id"] = run_id
        row["artifact_size_bytes"] = artifact_size
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    comparison.to_csv(output_dir / "experiment_comparison.csv", index=False)
    selected_row = comparison.iloc[0]
    selected_spec = next(spec for spec in _experiment_specs() if spec.name == selected_row["name"])
    LOGGER.info("Selected %s from CV; evaluating held-out test once", selected_spec.name)
    version = f"{__version__}+{(git_commit or 'nogit')[:8]}"

    with mlflow.start_run(run_name=f"final_{selected_spec.name}") as final_run:
        mlflow.log_params(
            {**shared_params, **selected_spec.__dict__, "selection_metric": "macro_f1"}
        )
        model = SentimentModel(build_pipeline(selected_spec, model_config), version=version)
        model.fit(train_reviews, train_labels)
        predictions = model.predict(test_reviews)
        probabilities = model.predict_proba(test_reviews)
        metrics = classification_metrics(test_labels, predictions, probabilities)
        metrics["by_language"] = _metrics_by_language(
            test_languages, test_labels, predictions, probabilities
        )
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if isinstance(value, float)}
        )
        model_path = output_dir / "model.joblib"
        model.save(model_path)
        model_hash = sha256_file(model_path)
        model_size = model_path.stat().st_size

        misclassified = []
        for review, actual, predicted, language in zip(
            test_reviews, test_labels, predictions, test_languages, strict=True
        ):
            if actual != predicted:
                misclassified.append(
                    {
                        "actual": actual,
                        "predicted": predicted,
                        "detected_language": language,
                        "excerpt": " ".join(review.split())[:160],
                    }
                )
        pd.DataFrame(misclassified).to_csv(output_dir / "error_analysis.csv", index=False)
        split_metadata = {
            "random_seed": seed,
            "test_size": float(config["test_size"]),
            "raw_rows": len(raw),
            "annotated_rows": len(annotated),
            "stratification_columns": ["detected_language", "Label"],
            "training_rows": len(split.train),
            "test_rows": len(split.test),
            "duplicate_rows_removed": split.duplicate_rows_removed,
            "conflicting_groups_removed": split.conflicting_groups_removed,
            "train_label_counts": split.train["Label"].value_counts().to_dict(),
            "test_label_counts": split.test["Label"].value_counts().to_dict(),
            "train_language_counts": split.train["detected_language"].value_counts().to_dict(),
            "test_language_counts": split.test["detected_language"].value_counts().to_dict(),
            "train_language_label_counts": {
                f"{language}::{label}": int(count)
                for (language, label), count in split.train.groupby(["detected_language", "Label"])
                .size()
                .items()
            },
            "test_language_label_counts": {
                f"{language}::{label}": int(count)
                for (language, label), count in split.test.groupby(["detected_language", "Label"])
                .size()
                .items()
            },
            "train_normalized_sha256": _hash_values(split.train["normalized_review"].tolist()),
            "test_normalized_sha256": _hash_values(split.test["normalized_review"].tolist()),
        }
        metadata = {
            "package_version": __version__,
            "model_version": version,
            "training_timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "raw_data_sha256": raw_hash,
            "model_sha256": model_hash,
            "model_size_bytes": model_size,
            "mlflow_run_id": final_run.info.run_id,
            "label_classes": list(model.pipeline.named_steps["classifier"].classes_),
            "expected_input_schema": {
                "review": (
                    f"non-empty Dutch or English string <= {MAX_REVIEW_CHARACTERS} characters"
                )
            },
            "language_configuration": config["language"],
            "split": split_metadata,
            "experiment": selected_spec.__dict__,
            "held_out_metrics": metrics,
        }
        (output_dir / "final_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (output_dir / "split_metadata.json").write_text(
            json.dumps(split_metadata, indent=2), encoding="utf-8"
        )
        (output_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        language_summary = pd.crosstab(
            language_evidence["Label"], language_evidence["language_status"]
        )
        language_summary.to_csv(output_dir / "language_summary.csv")
        mlflow.log_artifacts(str(output_dir), artifact_path="evidence")

    build_model_report(output_dir, report_dir / "model_report.md")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metadata = run_training(args.config)
    LOGGER.info("Trained model %s", metadata["model_version"])


if __name__ == "__main__":
    main()
