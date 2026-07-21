"""Run a staged, train-only RobBERT sweep before one final mixed-language test evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..constants import LABELS
from ..metrics import classification_metrics
from ..models.ordinal import with_ordinal_diagnostics
from ..models.robbert import (
    labels_from_probabilities,
    probability_rows,
    save_robbert_artifact,
    softmax_probabilities,
)
from ..text import normalize_text
from .common import language_slices
from .data import FrozenExperimentData, prepare_frozen_experiment
from .robbert_finetune import (
    _device_name,
    _git_commit,
    _predict_logits,
    _sha256_file,
    _train_model,
)

LOGGER = logging.getLogger(__name__)


def _candidate_id(candidate: dict[str, Any]) -> str:
    """Return a stable, filesystem-safe identifier for one candidate configuration."""
    return str(candidate["id"])


def _candidate_settings(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Overlay candidate input and loss choices on shared optimizer settings."""
    settings = deepcopy(base)
    settings.update({key: value for key, value in candidate.items() if key != "id"})
    return settings


def _normal_reviews(values: list[str]) -> list[str]:
    """Normalize reviews exactly as the serving and earlier experiment paths do."""
    return [normalize_text(value) for value in values]


def _evaluate_probabilities(
    labels: list[str], languages: list[str], probabilities: np.ndarray
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate shared classification, language, and ordinal diagnostics."""
    predictions = labels_from_probabilities(probabilities)
    metrics = with_ordinal_diagnostics(
        classification_metrics(labels, predictions, probability_rows(probabilities)),
        labels,
        predictions,
    )
    metrics["by_language"] = language_slices(labels, predictions, languages)
    return metrics, predictions


def _average_recall(metrics: dict[str, Any]) -> float:
    """Read Average recall from the shared per-class metric structure."""
    return float(metrics["per_class"]["Average"]["recall"])


def _trial_path(root: Path, candidate_id: str, fold: int, seed: int) -> Path:
    """Locate one resumable cross-validation result."""
    return root / "trials" / candidate_id / f"fold-{fold}-seed-{seed}.json"


def _run_trial(
    *,
    candidate: dict[str, Any],
    fold: int,
    seed: int,
    prepared: FrozenExperimentData,
    reviews: list[str],
    tokenizer: Any,
    config: dict[str, Any],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train or resume one fold/seed trial without touching the frozen test labels."""
    candidate_id = _candidate_id(candidate)
    destination = _trial_path(output_dir, candidate_id, fold, seed)
    if destination.is_file():
        return json.loads(destination.read_text(encoding="utf-8"))
    train_indices, validation_indices = prepared.folds[fold]
    settings = _candidate_settings(config["training"], candidate)
    started = time.perf_counter()
    model, best_epoch, history = _train_model(
        objective="multiclass_logistic",
        train_texts=[reviews[index] for index in train_indices],
        train_labels=[prepared.labels[index] for index in train_indices],
        validation_texts=[reviews[index] for index in validation_indices],
        validation_labels=[prepared.labels[index] for index in validation_indices],
        validation_languages=[prepared.languages[index] for index in validation_indices],
        tokenizer=tokenizer,
        model_settings=config["model"],
        training_settings=settings,
        seed=seed,
        epochs=int(settings["max_epochs"]),
        device=device,
    )
    logits = _predict_logits(
        model,
        tokenizer,
        [reviews[index] for index in validation_indices],
        [prepared.labels[index] for index in validation_indices],
        "multiclass_logistic",
        settings,
        device,
    )
    probabilities = softmax_probabilities(logits)
    metrics, predictions = _evaluate_probabilities(
        [prepared.labels[index] for index in validation_indices],
        [prepared.languages[index] for index in validation_indices],
        probabilities,
    )
    del model
    result = {
        "candidate_id": candidate_id,
        "candidate": candidate,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "validation_indices": validation_indices.tolist(),
        "predictions": predictions,
        "probabilities": probabilities.tolist(),
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _summary(candidate: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize fold/seed scores without pooling test information."""
    macro_f1 = [float(trial["metrics"]["macro_f1"]) for trial in trials]
    accuracy = [float(trial["metrics"]["accuracy"]) for trial in trials]
    average_recall = [_average_recall(trial["metrics"]) for trial in trials]
    return {
        "candidate_id": _candidate_id(candidate),
        "candidate": candidate,
        "runs": len(trials),
        "macro_f1_mean": statistics.fmean(macro_f1),
        "macro_f1_std": statistics.pstdev(macro_f1),
        "accuracy_mean": statistics.fmean(accuracy),
        "average_recall_mean": statistics.fmean(average_recall),
        "best_epochs": [int(trial["best_epoch"]) for trial in trials],
    }


def _select_screened(
    summaries: list[dict[str, Any]], top_k: int, minimum_average_recall: float
) -> list[str]:
    """Promote stable screen candidates while guarding against Average-class collapse."""
    eligible = [row for row in summaries if row["average_recall_mean"] >= minimum_average_recall]
    ranked = sorted(eligible or summaries, key=lambda row: row["macro_f1_mean"], reverse=True)
    return [str(row["candidate_id"]) for row in ranked[:top_k]]


def _aggregate_oof(
    candidate: dict[str, Any], trials: list[dict[str, Any]], prepared: FrozenExperimentData
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Average repeated-fold predictions per row and compute honest train-only OOF metrics."""
    probability_sum = np.zeros((prepared.train_rows, len(LABELS)), dtype=float)
    counts = np.zeros(prepared.train_rows, dtype=int)
    for trial in trials:
        indices = np.asarray(trial["validation_indices"], dtype=int)
        probability_sum[indices] += np.asarray(trial["probabilities"], dtype=float)
        counts[indices] += 1
    if np.any(counts == 0):
        raise RuntimeError(f"Incomplete OOF coverage for {_candidate_id(candidate)}")
    probabilities = probability_sum / counts.reshape(-1, 1)
    metrics, predictions = _evaluate_probabilities(
        prepared.labels, prepared.languages, probabilities
    )
    frame = pd.DataFrame(
        {
            "candidate_id": _candidate_id(candidate),
            "row_index": range(prepared.train_rows),
            "actual": prepared.labels,
            "detected_language": prepared.languages,
            "prediction": predictions,
        }
    )
    for index, label in enumerate(LABELS):
        frame[f"p_{label.lower()}"] = probabilities[:, index]
    return metrics, frame


def _final_fit(
    *,
    candidate: dict[str, Any],
    selected_epochs: int,
    prepared: FrozenExperimentData,
    reviews: list[str],
    test_reviews: list[str],
    tokenizer: Any,
    config: dict[str, Any],
    output_dir: Path,
    device: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Refit the frozen candidate for each seed and ensemble one final test evaluation."""
    settings = _candidate_settings(config["training"], candidate)
    seed_probabilities: list[np.ndarray] = []
    fit_rows: list[dict[str, Any]] = []
    for seed in config["confirmation"]["seeds"]:
        started = time.perf_counter()
        model, _, history = _train_model(
            objective="multiclass_logistic",
            train_texts=reviews,
            train_labels=prepared.labels,
            validation_texts=None,
            validation_labels=None,
            validation_languages=None,
            tokenizer=tokenizer,
            model_settings=config["model"],
            training_settings=settings,
            seed=int(seed),
            epochs=selected_epochs,
            device=device,
        )
        logits = _predict_logits(
            model,
            tokenizer,
            test_reviews,
            prepared.test_labels,
            "multiclass_logistic",
            settings,
            device,
        )
        seed_probabilities.append(softmax_probabilities(logits))
        save_robbert_artifact(
            model,
            tokenizer,
            output_dir / "models" / _candidate_id(candidate) / f"seed-{seed}",
            model_id=config["model"]["model_id"],
            revision=config["model"]["revision"],
            objective="multiclass_logistic",
            max_length=int(settings["max_length"]),
            input_strategy=str(settings["input_strategy"]),
            loss=str(settings["loss"]),
        )
        del model
        fit_rows.append(
            {
                "seed": int(seed),
                "epochs": selected_epochs,
                "history": history,
                "seconds": time.perf_counter() - started,
            }
        )
    probabilities = np.mean(seed_probabilities, axis=0)
    metrics, predictions = _evaluate_probabilities(
        prepared.test_labels, prepared.test_languages, probabilities
    )
    frame = pd.DataFrame(
        {
            "row_index": range(prepared.heldout_rows),
            "actual": prepared.test_labels,
            "detected_language": prepared.test_languages,
            "prediction": predictions,
        }
    )
    for index, label in enumerate(LABELS):
        frame[f"p_{label.lower()}"] = probabilities[:, index]
    return {"metrics": metrics, "fits": fit_rows}, frame


def _write_checksums(output_dir: Path) -> None:
    """Checksum portable evidence and final model files, excluding resumable trial cache."""
    material = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and "trials" not in path.parts and path.name != "checksums.sha256"
    ]
    lines = [f"{_sha256_file(path)}  {path.relative_to(output_dir)}" for path in sorted(material)]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trial_index(output_dir: str | Path) -> Path:
    """Summarize resumable trial files into one compact review-friendly JSONL index."""
    root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "trials").glob("*/fold-*-seed-*.json")):
        trial = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "candidate_id": str(trial["candidate_id"]),
                "fold": int(trial["fold"]),
                "seed": int(trial["seed"]),
                "best_epoch": int(trial["best_epoch"]),
                "macro_f1": float(trial["metrics"]["macro_f1"]),
                "accuracy": float(trial["metrics"]["accuracy"]),
                "average_recall": _average_recall(trial["metrics"]),
                "seconds": float(trial["seconds"]),
            }
        )
    destination = root / "trials_index.jsonl"
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return destination


def run_experiment(
    config_path: str | Path, output_override: str | Path | None = None
) -> dict[str, Any]:
    """Screen candidates, confirm finalists with repeated CV, and evaluate one test ensemble."""
    import torch
    import transformers
    from transformers import AutoTokenizer

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    prepared = prepare_frozen_experiment(config)
    output_dir = Path(output_override or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_name(str(config["training"].get("device", "auto")))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_id"], revision=config["model"]["revision"], use_fast=True
    )
    reviews = _normal_reviews(prepared.reviews)
    test_reviews = _normal_reviews(prepared.test_reviews)
    candidates = {str(row["id"]): row for row in config["screening"]["candidates"]}

    screen_trials = [
        _run_trial(
            candidate=candidate,
            fold=int(config["screening"]["fold"]),
            seed=int(config["screening"]["seed"]),
            prepared=prepared,
            reviews=reviews,
            tokenizer=tokenizer,
            config=config,
            output_dir=output_dir,
            device=device,
        )
        for candidate in candidates.values()
    ]
    screen_summaries = [
        _summary(candidate, [trial])
        for candidate, trial in zip(candidates.values(), screen_trials, strict=True)
    ]
    promoted_ids = _select_screened(
        screen_summaries,
        int(config["screening"]["promote_top_k"]),
        float(config["screening"]["minimum_average_recall"]),
    )

    confirmation_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    confirmation_trials: dict[str, list[dict[str, Any]]] = {}
    for candidate_id in promoted_ids:
        candidate = candidates[candidate_id]
        trials = [
            _run_trial(
                candidate=candidate,
                fold=fold,
                seed=int(seed),
                prepared=prepared,
                reviews=reviews,
                tokenizer=tokenizer,
                config=config,
                output_dir=output_dir,
                device=device,
            )
            for fold in range(int(config["confirmation"]["folds"]))
            for seed in config["confirmation"]["seeds"]
        ]
        confirmation_trials[candidate_id] = trials
        summary = _summary(candidate, trials)
        oof_metrics, oof_frame = _aggregate_oof(candidate, trials, prepared)
        summary["oof_metrics"] = oof_metrics
        confirmation_rows.append(summary)
        oof_frames.append(oof_frame)
    confirmation_rows.sort(key=lambda row: row["oof_metrics"]["macro_f1"], reverse=True)
    winner_id = str(confirmation_rows[0]["candidate_id"])
    winner = candidates[winner_id]

    final_result: dict[str, Any] | None = None
    if bool(config.get("evaluate_test", False)):
        epochs = max(
            1,
            round(
                statistics.median(trial["best_epoch"] for trial in confirmation_trials[winner_id])
            ),
        )
        final_result, test_frame = _final_fit(
            candidate=winner,
            selected_epochs=epochs,
            prepared=prepared,
            reviews=reviews,
            test_reviews=test_reviews,
            tokenizer=tokenizer,
            config=config,
            output_dir=output_dir,
            device=device,
        )
        final_result.update({"candidate_id": winner_id, "selected_epochs": epochs})
        test_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_csv(output_dir / "oof_predictions.csv", index=False)

    result = {
        "experiment": "robbert-v2-mixed-language-improvement-v2",
        "source_commit": _git_commit(),
        "test_used_for_selection": False,
        "test_status": "previously_viewed_comparison_set",
        "data": {
            "raw_sha256": prepared.raw_sha256,
            "train_normalized_sha256": prepared.train_sha256,
            "test_normalized_sha256": prepared.heldout_sha256,
            "train_rows": prepared.train_rows,
            "test_rows": prepared.heldout_rows,
            "language_policy": "single_mixed_dutch_english_model",
        },
        "model": config["model"],
        "screening": screen_summaries,
        "promoted_candidates": promoted_ids,
        "confirmation": confirmation_rows,
        "winner": winner_id,
        "final_test": final_result,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_trial_index(output_dir)
    _write_checksums(output_dir)
    return result


def verify_bundle(
    bundle_dir: str | Path, config_path: str | Path, *, require_models: bool = True
) -> dict[str, Any]:
    """Verify evidence and optionally require the three large final model directories."""
    root = Path(bundle_dir)
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() and not require_models and relative.startswith("models/"):
            continue
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"Bundle checksum mismatch: {relative}")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    result = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    if result["data"]["train_normalized_sha256"] != config["expected_train_normalized_sha256"]:
        raise RuntimeError("Bundle training split hash does not match the frozen config")
    if result["data"]["test_normalized_sha256"] != config["expected_test_normalized_sha256"]:
        raise RuntimeError("Bundle test split hash does not match the frozen config")
    oof = pd.read_csv(root / "oof_predictions.csv")
    for candidate_id in result["promoted_candidates"]:
        if len(oof[oof["candidate_id"] == candidate_id]) != int(result["data"]["train_rows"]):
            raise RuntimeError(f"Incomplete OOF rows for {candidate_id}")
    if result["final_test"] is not None:
        test = pd.read_csv(root / "test_predictions.csv")
        if len(test) != int(result["data"]["test_rows"]):
            raise RuntimeError("Bundle test prediction row count does not match its manifest")
    return result


def _register_challenger(
    client: Any,
    run_id: str,
    result: dict[str, Any],
    config: dict[str, Any],
    root: Path,
) -> None:
    """Register the winner as a non-deployable challenger without changing Production."""
    name = "sentiment-robbert-v2-improved"
    weights_stored = all(
        (root / "models" / result["winner"] / f"seed-{seed}" / "model.safetensors").is_file()
        for seed in (42, 73, 101)
    )
    try:
        client.get_registered_model(name)
    except Exception:  # MLflow raises a store-specific not-found exception.
        client.create_registered_model(name)
    description = (
        "Mixed-language RobBERT v2 winner selected by staged train-only validation; "
        "presentation challenger evidence, not approved for Production."
    )
    client.update_registered_model(name, description=description)
    tags = {
        "governance.role": "challenger-evaluation",
        "deployment.eligible": "false",
        "presentation.selected": "false",
        "training.strategy": "end-to-end-finetuning",
        "language.policy": "mixed-dutch-english",
        "license": str(config["model"]["license"]),
        "artifact.tier": "deployable" if weights_stored else "evidence-only",
        "artifact.weights_stored": str(weights_stored).lower(),
        "artifact.remote_bundle": "google-drive",
    }
    for key, value in tags.items():
        client.set_registered_model_tag(name, key, value)
    source = (
        f"runs:/{run_id}/models/{result['winner']}"
        if weights_stored
        else f"runs:/{run_id}/evidence"
    )
    versions = client.search_model_versions(f"name = '{name}'")
    version = next(
        (item for item in versions if str(item.run_id) == run_id and str(item.source) == source),
        None,
    )
    if version is None:
        version = client.create_model_version(name=name, source=source, run_id=run_id)
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version.version), key, value)
    client.set_registered_model_alias(name, "challenger-evaluation", str(version.version))


def log_existing_bundle(bundle_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Import verified improvement evidence into local MLflow without retraining."""
    import mlflow
    from mlflow import MlflowClient

    root = Path(bundle_dir)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    result = verify_bundle(root, config_path, require_models=False)
    tracking_uri = str(config["mlflow_tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(config["mlflow_experiment"])
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(
            config["mlflow_experiment"], tags={"purpose": "robbert-improvement"}
        )
    )
    matches = client.search_runs(
        [experiment_id], filter_string=f"tags.catalog_id = '{config['mlflow_catalog_id']}'"
    )
    if matches:
        run_id = str(matches[0].info.run_id)
        _register_challenger(client, run_id, result, config, root)
        result["mlflow_run_id"] = run_id
        return result
    weights_stored = (root / "models").is_dir()
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"robbert-improvement-{result['winner']}",
    ) as run:
        mlflow.set_tags(
            {
                "catalog_id": config["mlflow_catalog_id"],
                "governance.role": "challenger-evaluation",
                "artifact.tier": "deployable" if weights_stored else "evidence-only",
                "artifact.weights_stored": str(weights_stored).lower(),
                "artifact.remote_bundle": "google-drive",
                "deployment.eligible": "false",
                "presentation.selected": "false",
                "production_champion_changed": "false",
                "language.policy": "mixed-dutch-english",
                "test_used_for_selection": "false",
                "test.status": "previously-viewed",
                "source.commit": result["source_commit"],
            }
        )
        for row in result["confirmation"]:
            prefix = str(row["candidate_id"])
            mlflow.log_metric(f"{prefix}.oof_macro_f1", row["oof_metrics"]["macro_f1"])
            mlflow.log_metric(f"{prefix}.cv_macro_f1_std", row["macro_f1_std"])
        if result["final_test"] is not None:
            for key, value in result["final_test"]["metrics"].items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    mlflow.log_metric(f"winner.test_{key}", float(value))
        if weights_stored:
            mlflow.log_artifacts(str(root / "models"), artifact_path="models")
        mlflow.log_artifacts(str(root), artifact_path="evidence")
        run_id = str(run.info.run_id)
        result["mlflow_run_id"] = run_id
    _register_challenger(client, run_id, result, config, root)
    return result


def main() -> None:
    """Run, verify, or import the mixed-language RobBERT improvement experiment."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/robbert_improvement.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--verify-bundle")
    parser.add_argument("--log-existing")
    args = parser.parse_args()
    if args.verify_bundle:
        result = verify_bundle(args.verify_bundle, args.config)
    elif args.log_existing:
        result = log_existing_bundle(args.log_existing, args.config)
    else:
        result = run_experiment(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
