"""Evaluate the five frozen presentation models on the same reused held-out split."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from .config import load_config
from .constants import LABELS
from .data import annotate_review_languages, load_dataset, make_holdout_split, sha256_file
from .experiments.common import aligned_probabilities, hash_values
from .experiments.jina_ordinal import _calibrated_boundary_classifier
from .language import DutchLanguageDetector
from .metrics import classification_metrics
from .model import SentimentModel, load_sentiment_model
from .models.embeddings import encode_or_load
from .models.ordinal import (
    ORDERED_LABELS,
    compose_ordinal_probabilities,
    probability_argmax_labels,
    project_monotonic_boundaries,
    with_ordinal_diagnostics,
)
from .text import normalize_text

MODEL_ORDER = (
    "Current Production TF-IDF",
    "TF-IDF Ordinal",
    "Jina Logistic",
    "Jina Ordinal",
    "DeepSeek V4 Flash 24-shot",
)


def _restore_loaded_logistic_state(estimator: Any) -> Any:
    """Restore removed sklearn state required by older serialized multiclass heads."""
    candidates = [estimator]
    # Avoid get_params(): newer sklearn asks the legacy child for the missing
    # field before we have a chance to repair it.
    candidates.extend(component for _, component in getattr(estimator, "steps", ()))
    for candidate in candidates:
        if not isinstance(candidate, LogisticRegression):
            continue
        classes = getattr(candidate, "classes_", ())
        if len(classes) > 2 and not hasattr(candidate, "multi_class"):
            # The frozen 1.9 artifact omitted this retired constructor field, while
            # some runtimes still consult it when computing multiclass probabilities.
            candidate.multi_class = "multinomial"
    return estimator


def _git_bytes(reference: str, path: str) -> bytes:
    """Read a required immutable artifact from Git without changing the worktree."""
    result = subprocess.run(
        ["git", "show", f"{reference}:{path}"], capture_output=True, check=False
    )
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Cannot read {reference}:{path}: {error}")
    return result.stdout


def _review_sha256(review: str) -> str:
    """Return the row identity used by the archived DeepSeek experiment."""
    return hashlib.sha256(normalize_text(review).encode("utf-8")).hexdigest()


def _prepare_split(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recreate the frozen split and fail closed if either row sequence changed."""
    training = load_config(config["training_config"])
    raw = load_dataset(training["data_path"])
    detector = DutchLanguageDetector(**training["language"])
    annotated, _ = annotate_review_languages(raw, detector)
    split = make_holdout_split(
        annotated,
        test_size=float(training["test_size"]),
        random_seed=int(training["random_seed"]),
        stratify_columns=("detected_language", "Label"),
    )
    train_hash = hash_values(split.train["normalized_review"].tolist())
    test_hash = hash_values(split.test["normalized_review"].tolist())
    if train_hash != config["expected_train_normalized_sha256"]:
        raise RuntimeError("Frozen training split hash changed; refusing final comparison")
    if test_hash != config["expected_test_normalized_sha256"]:
        raise RuntimeError("Frozen held-out split hash changed; refusing final comparison")
    provenance = {
        "raw_sha256": sha256_file(training["data_path"]),
        "train_normalized_sha256": train_hash,
        "heldout_normalized_sha256": test_hash,
        "train_rows": len(split.train),
        "heldout_rows": len(split.test),
        "random_seed": int(training["random_seed"]),
    }
    return split.train, split.test, provenance


def _load_archived_model(reference: str, path: str) -> Any:
    """Materialize and load a trusted model stored in an immutable Git archive."""
    payload = _git_bytes(reference, path)
    with tempfile.NamedTemporaryFile(suffix=".joblib") as handle:
        handle.write(payload)
        handle.flush()
        return load_sentiment_model(handle.name)


def _probability_rows(matrix: np.ndarray, labels: tuple[str, ...]) -> list[dict[str, float]]:
    """Convert an aligned probability matrix into metric-compatible dictionaries."""
    return [{label: float(row[index]) for index, label in enumerate(labels)} for row in matrix]


def _jina_predictions(
    train: pd.DataFrame, heldout: pd.DataFrame, config: dict[str, Any], output_dir: Path
) -> tuple[dict[str, tuple[list[str], list[dict[str, float]]]], dict[str, Any]]:
    """Fit the two frozen Jina heads on train embeddings and predict held-out once."""
    settings = config["jina"]
    reviews = (
        pd.concat([train["Reviews"], heldout["Reviews"]], ignore_index=True).astype(str).tolist()
    )
    embeddings, runtime = encode_or_load(settings["model"], reviews, settings)
    train_embeddings = embeddings[: len(train)]
    heldout_embeddings = embeddings[len(train) :]
    labels = train["Label"].astype(str).to_numpy()
    seed = int(config["random_seed"])

    logistic_settings = settings["logistic"]
    logistic = LogisticRegression(
        C=float(logistic_settings["C"]),
        class_weight=logistic_settings["class_weight"],
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    logistic.fit(train_embeddings, labels)
    logistic_probabilities = aligned_probabilities(logistic, heldout_embeddings)
    logistic_predictions = [LABELS[index] for index in logistic_probabilities.argmax(axis=1)]

    ordinal_settings = settings["ordinal"]
    lower = _calibrated_boundary_classifier(
        c_value=float(ordinal_settings["C"]),
        seed=seed,
        calibration_folds=int(ordinal_settings["calibration_folds"]),
    )
    upper = _calibrated_boundary_classifier(
        c_value=float(ordinal_settings["C"]),
        seed=seed,
        calibration_folds=int(ordinal_settings["calibration_folds"]),
    )
    lower.fit(train_embeddings, (labels != "Negative").astype(int))
    upper.fit(train_embeddings, (labels == "Positive").astype(int))
    above_negative, above_average, projection = project_monotonic_boundaries(
        lower.predict_proba(heldout_embeddings)[:, 1],
        upper.predict_proba(heldout_embeddings)[:, 1],
    )
    ordinal_probabilities = compose_ordinal_probabilities(above_negative, above_average)
    ordinal_predictions = probability_argmax_labels(ordinal_probabilities)

    heads_path = output_dir / "jina_frozen_heads.joblib"
    joblib.dump(
        {
            "embedding_model": settings["model"],
            "logistic": logistic,
            "ordinal_lower": lower,
            "ordinal_upper": upper,
            "label_order": list(LABELS),
            "ordinal_label_order": list(ORDERED_LABELS),
        },
        heads_path,
        compress=3,
    )
    runtime.update(
        {
            "heads_artifact": str(heads_path),
            "heads_sha256": sha256_file(heads_path),
            "ordinal_projection": projection,
        }
    )
    return (
        {
            "Jina Logistic": (
                logistic_predictions,
                _probability_rows(logistic_probabilities, LABELS),
            ),
            "Jina Ordinal": (
                ordinal_predictions,
                _probability_rows(ordinal_probabilities, ORDERED_LABELS),
            ),
        },
        runtime,
    )


def _deepseek_predictions(heldout: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Align archived DeepSeek labels to held-out rows using two independent keys."""
    payload = _git_bytes(config["deepseek_archive"], config["deepseek_predictions"])
    archived = pd.read_csv(io.BytesIO(payload))
    archived = archived.loc[(archived["split"] == "heldout") & (archived["status"] == "ok")]
    if len(archived) != len(heldout):
        raise RuntimeError(f"Expected {len(heldout)} DeepSeek held-out rows, found {len(archived)}")
    indexed = archived.set_index("source_row", verify_integrity=True)
    predictions: list[str] = []
    for row in heldout.itertuples(index=False):
        match = indexed.loc[int(row.source_row)]
        expected_hash = _review_sha256(str(row.Reviews))
        if str(match.review_sha256) != expected_hash:
            raise RuntimeError(f"DeepSeek review hash mismatch at source row {row.source_row}")
        if str(match.actual) != str(row.Label):
            raise RuntimeError(f"DeepSeek actual label mismatch at source row {row.source_row}")
        label = str(match.label)
        if label not in LABELS:
            raise RuntimeError(f"DeepSeek returned invalid label {label!r}")
        predictions.append(label)
    return predictions


def _metrics(
    labels: list[str],
    predictions: list[str],
    languages: list[str],
    probabilities: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Calculate common classification, ordinal, and language-slice evidence."""
    result = with_ordinal_diagnostics(
        classification_metrics(labels, predictions, probabilities), labels, predictions
    )
    result["by_language"] = {}
    for language in sorted(set(languages)):
        indices = [index for index, value in enumerate(languages) if value == language]
        result["by_language"][language] = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
    return result


def _summary_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten the comparison metrics needed for ranking and presentation."""
    negative = metrics["per_class"]["Negative"]
    return {
        "model": name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "negative_precision": negative["precision"],
        "negative_recall": negative["recall"],
        "negative_f1": negative["f1-score"],
        "ordinal_mae": metrics["ordinal_mae"],
        "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
        "severe_error_rate": metrics["severe_error_rate"],
        "log_loss": metrics.get("log_loss"),
        "multiclass_brier_score": metrics.get("multiclass_brier_score"),
        "expected_calibration_error_10_bin": metrics.get("expected_calibration_error_10_bin"),
    }


def _write_report(summary: pd.DataFrame, result: dict[str, Any], path: Path) -> None:
    """Write an explicit final-presentation comparison report."""
    table = summary.copy()
    numeric = table.select_dtypes(include="number").columns
    table[numeric] = table[numeric].round(4)
    content = f"""# Final five-model comparison

## Interpretation

All five frozen candidates were evaluated against the same {result["data"]["heldout_rows"]}-row
held-out split. The split was never used to fit any of the five models, but it has appeared in prior
project reports. Therefore this is a **reused-heldout presentation comparison**, not a new blind test
and not authorization to tune parameters after seeing the ranking.

Ranking metric: held-out Macro-F1, descending.

{table.to_markdown(index=False)}

## Governance

- Selected for presentation: exactly the five rows above.
- Production remains `Current Production TF-IDF`; this comparison does not change the champion.
- Jina models are research-only under CC-BY-NC-4.0 and use a pinned external encoder.
- DeepSeek is an archived external API result; provider weights are not stored locally.
- All other registered models remain benchmark, ablation, baseline, or test-only evidence.
- `parameters_frozen_before_evaluation=true`; `post_evaluation_tuning_allowed=false`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _log_mlflow(result: dict[str, Any], output_dir: Path, config: dict[str, Any]) -> str:
    """Log one idempotent evidence run containing the complete five-model result."""
    tracking_uri = str(config["mlflow_tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(config["mlflow_experiment"])
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(
            config["mlflow_experiment"], tags={"purpose": "final-presentation-comparison"}
        )
    )
    matches = client.search_runs(
        [experiment_id], filter_string=f"tags.catalog_id = '{config['mlflow_catalog_id']}'"
    )
    if matches:
        return str(matches[0].info.run_id)
    tags = {
        "catalog_id": config["mlflow_catalog_id"],
        "evaluation.scope": "reused-heldout-presentation-comparison",
        "parameters_frozen_before_evaluation": "true",
        "post_evaluation_tuning_allowed": "false",
        "production_champion_changed": "false",
    }
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="final-five-heldout", tags=tags
    ) as run:
        for row in result["ranking"]:
            prefix = row["model"].lower().replace(" ", "_").replace("-", "_")
            for key, value in row.items():
                if key != "model" and value is not None:
                    mlflow.log_metric(f"{prefix}.{key}", float(value))
        mlflow.log_artifacts(str(output_dir), artifact_path="final_comparison")
        return str(run.info.run_id)


def run_comparison(config_path: str | Path, *, log_mlflow: bool = False) -> dict[str, Any]:
    """Run the frozen comparison, persist predictions/metrics, and optionally log MLflow."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    train, heldout, provenance = _prepare_split(config)
    config["random_seed"] = provenance["random_seed"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reviews = heldout["Reviews"].astype(str).tolist()
    labels = heldout["Label"].astype(str).tolist()
    languages = heldout["detected_language"].astype(str).tolist()

    production = SentimentModel.load(config["production_artifact"])
    _restore_loaded_logistic_state(production.pipeline)
    ordinal = _load_archived_model(config["ordinal_archive"], config["ordinal_artifact"])
    predictions: dict[str, tuple[list[str], list[dict[str, float]] | None]] = {
        "Current Production TF-IDF": (
            production.predict(reviews),
            production.predict_proba(reviews),
        ),
        "TF-IDF Ordinal": (ordinal.predict(reviews), ordinal.predict_proba(reviews)),
    }
    jina, jina_runtime = _jina_predictions(train, heldout, config, output_dir)
    predictions.update(jina)
    predictions["DeepSeek V4 Flash 24-shot"] = (_deepseek_predictions(heldout, config), None)

    metrics = {
        name: _metrics(labels, model_predictions, languages, probabilities)
        for name, (model_predictions, probabilities) in predictions.items()
    }
    summary = pd.DataFrame([_summary_row(name, metrics[name]) for name in MODEL_ORDER])
    summary = summary.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    prediction_frame = heldout[["source_row", "Reviews", "Label", "detected_language"]].copy()
    prediction_frame["review_sha256"] = prediction_frame["Reviews"].map(_review_sha256)
    for name in MODEL_ORDER:
        prediction_frame[name] = predictions[name][0]

    result = {
        "evaluation_scope": "reused-heldout-presentation-comparison",
        "parameters_frozen_before_evaluation": True,
        "post_evaluation_tuning_allowed": False,
        "production_champion_changed": False,
        "selected_models": list(MODEL_ORDER),
        "data": provenance,
        "ranking": summary.where(pd.notna(summary), None).to_dict(orient="records"),
        "metrics": metrics,
        "jina_runtime": jina_runtime,
        "sources": {
            "tfidf_ordinal": f"{config['ordinal_archive']}:{config['ordinal_artifact']}",
            "deepseek": f"{config['deepseek_archive']}:{config['deepseek_predictions']}",
        },
    }
    summary.to_csv(output_dir / "comparison.csv", index=False)
    prediction_frame.to_csv(output_dir / "heldout_predictions.csv", index=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(summary, result, Path(config["report_path"]))
    if log_mlflow:
        result["mlflow_run_id"] = _log_mlflow(result, output_dir, config)
        (output_dir / "comparison.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    return result


def main() -> None:
    """Run the final five-model comparison CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/final_five_comparison.yaml")
    parser.add_argument("--log-mlflow", action="store_true")
    args = parser.parse_args()
    result = run_comparison(args.config, log_mlflow=args.log_mlflow)
    print(json.dumps(result["ranking"], indent=2))


if __name__ == "__main__":
    main()
