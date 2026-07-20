"""Fine-tune paired RobBERT multiclass-logistic and CORAL-ordinal candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..constants import LABELS
from ..metrics import classification_metrics
from ..models.ordinal import with_ordinal_diagnostics
from ..models.robbert import (
    OBJECTIVES,
    balanced_boundary_weights,
    balanced_class_weights,
    build_robbert_model,
    coral_probabilities,
    labels_from_probabilities,
    multiclass_targets,
    ordinal_targets,
    probability_rows,
    save_robbert_artifact,
    softmax_probabilities,
)
from ..text import normalize_text
from .common import language_slices
from .data import FrozenExperimentData, prepare_frozen_experiment

LOGGER = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    """Hash one result file in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    """Return the source commit when the experiment runs from a Git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unavailable"


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without requiring CUDA."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_name(requested: str) -> str:
    """Resolve an explicit or automatic PyTorch execution device."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available")
    return requested


def _collator(tokenizer: Any, max_length: int) -> Any:
    """Build a dynamic-padding collator that keeps task targets separate."""
    import torch

    def collate(batch: list[tuple[str, np.ndarray | np.int64]]) -> tuple[dict[str, Any], Any]:
        texts, targets = zip(*batch, strict=True)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded, torch.as_tensor(np.asarray(targets))

    return collate


def _dataset(texts: list[str], labels: list[str], objective: str) -> Any:
    """Create a small in-memory dataset with objective-specific targets."""
    from torch.utils.data import Dataset

    targets = (
        multiclass_targets(labels)
        if objective == "multiclass_logistic"
        else ordinal_targets(labels)
    )

    class ReviewDataset(Dataset):
        """Expose normalized review strings and fixed targets to a DataLoader."""

        def __len__(self) -> int:
            return len(texts)

        def __getitem__(self, index: int) -> tuple[str, Any]:
            return texts[index], targets[index]

    return ReviewDataset()


def _loss_function(objective: str, training_labels: list[str], device: str) -> Any:
    """Build training-only balanced loss weights for one objective."""
    import torch
    from torch.nn import functional as torch_functional

    if objective == "multiclass_logistic":
        weights = torch.tensor(balanced_class_weights(training_labels), device=device).float()

        def loss(logits: Any, targets: Any) -> Any:
            return torch_functional.cross_entropy(logits, targets.long(), weight=weights)

        return loss

    boundary_weights = torch.tensor(
        balanced_boundary_weights(training_labels), device=device
    ).float()

    def ordinal_loss(logits: Any, targets: Any) -> Any:
        values = targets.to(device=device, dtype=torch.float32)
        indices = values.long().transpose(0, 1)
        sample_weights = torch.stack(
            [boundary_weights[boundary, indices[boundary]] for boundary in range(2)], dim=1
        )
        return torch_functional.binary_cross_entropy_with_logits(
            logits, values, weight=sample_weights
        )

    return ordinal_loss


def _predict_logits(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    labels: list[str],
    objective: str,
    settings: dict[str, Any],
    device: str,
) -> np.ndarray:
    """Run deterministic batched evaluation and return CPU logits."""
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        _dataset(texts, labels, objective),
        batch_size=int(settings["evaluation_batch_size"]),
        shuffle=False,
        collate_fn=_collator(tokenizer, int(settings["max_length"])),
    )
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            rows.append(model(**inputs).detach().float().cpu().numpy())
    return np.concatenate(rows)


def _probabilities(objective: str, logits: np.ndarray) -> np.ndarray:
    """Convert objective logits to the shared project probability order."""
    return (
        softmax_probabilities(logits)
        if objective == "multiclass_logistic"
        else coral_probabilities(logits)
    )


def _evaluate(
    objective: str,
    labels: list[str],
    languages: list[str],
    logits: np.ndarray,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    """Calculate shared classification, probability, language, and ordinal evidence."""
    probabilities = _probabilities(objective, logits)
    predictions = labels_from_probabilities(probabilities)
    metrics = with_ordinal_diagnostics(
        classification_metrics(labels, predictions, probability_rows(probabilities)),
        labels,
        predictions,
    )
    metrics["by_language"] = language_slices(labels, predictions, languages)
    return metrics, predictions, probabilities


def _train_model(
    *,
    objective: str,
    train_texts: list[str],
    train_labels: list[str],
    validation_texts: list[str] | None,
    validation_labels: list[str] | None,
    validation_languages: list[str] | None,
    tokenizer: Any,
    model_settings: dict[str, Any],
    training_settings: dict[str, Any],
    seed: int,
    epochs: int,
    device: str,
) -> tuple[Any, int, list[dict[str, Any]]]:
    """Fine-tune one RobBERT objective with optional train-only early stopping."""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    _set_seed(seed)
    model = build_robbert_model(
        model_settings["model_id"],
        model_settings["revision"],
        objective,
        dropout=float(model_settings.get("dropout", 0.1)),
    ).to(device)
    loader = DataLoader(
        _dataset(train_texts, train_labels, objective),
        batch_size=int(training_settings["batch_size"]),
        shuffle=True,
        collate_fn=_collator(tokenizer, int(training_settings["max_length"])),
    )
    accumulation = int(training_settings["gradient_accumulation_steps"])
    updates_per_epoch = max(1, (len(loader) + accumulation - 1) // accumulation)
    total_updates = updates_per_epoch * epochs
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_settings["learning_rate"]),
        weight_decay=float(training_settings["weight_decay"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * float(training_settings["warmup_ratio"])),
        num_training_steps=total_updates,
    )
    loss_function = _loss_function(objective, train_labels, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best_state: dict[str, Any] | None = None
    best_epoch = epochs
    best_macro_f1 = -1.0
    patience = 0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (inputs, targets) in enumerate(loader, start=1):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            targets = targets.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_function(model(**inputs), targets) / accumulation
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * accumulation
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        epoch_result: dict[str, Any] = {"epoch": epoch, "train_loss": total_loss / len(loader)}
        if validation_texts is not None:
            logits = _predict_logits(
                model,
                tokenizer,
                validation_texts,
                validation_labels or [],
                objective,
                training_settings,
                device,
            )
            metrics, _, _ = _evaluate(
                objective,
                validation_labels or [],
                validation_languages or ["unknown"] * len(validation_texts),
                logits,
            )
            epoch_result["validation"] = metrics
            macro_f1 = float(metrics["macro_f1"])
            if macro_f1 > best_macro_f1 + 1e-8:
                best_macro_f1 = macro_f1
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
        history.append(epoch_result)
        LOGGER.info("%s seed=%s epoch=%s result=%s", objective, seed, epoch, epoch_result)
        if validation_texts is not None and patience >= int(
            training_settings["early_stopping_patience"]
        ):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def _normalized_reviews(values: list[str]) -> list[str]:
    """Apply the serving-compatible normalization before tokenization and hashing."""
    return [normalize_text(value) for value in values]


def _run_candidate(
    objective: str,
    prepared: FrozenExperimentData,
    tokenizer: Any,
    config: dict[str, Any],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Select epochs on one frozen fold, refit all train rows, and evaluate test once."""
    settings = config["training"]
    train_indices, validation_indices = prepared.folds[int(settings["validation_fold"])]
    reviews = _normalized_reviews(prepared.reviews)
    test_reviews = _normalized_reviews(prepared.test_reviews)
    validation_runs: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    for seed in settings["validation_seeds"]:
        started = time.perf_counter()
        model, best_epoch, history = _train_model(
            objective=objective,
            train_texts=[reviews[index] for index in train_indices],
            train_labels=[prepared.labels[index] for index in train_indices],
            validation_texts=[reviews[index] for index in validation_indices],
            validation_labels=[prepared.labels[index] for index in validation_indices],
            validation_languages=[prepared.languages[index] for index in validation_indices],
            tokenizer=tokenizer,
            model_settings=config["model"],
            training_settings=settings,
            seed=int(seed),
            epochs=int(settings["max_epochs"]),
            device=device,
        )
        del model
        best_epochs.append(best_epoch)
        validation_runs.append(
            {
                "seed": int(seed),
                "best_epoch": best_epoch,
                "best_validation": history[best_epoch - 1]["validation"],
                "history": history,
                "seconds": time.perf_counter() - started,
            }
        )
    final_epochs = max(1, round(statistics.median(best_epochs)))
    started = time.perf_counter()
    final_model, _, final_history = _train_model(
        objective=objective,
        train_texts=reviews,
        train_labels=prepared.labels,
        validation_texts=None,
        validation_labels=None,
        validation_languages=None,
        tokenizer=tokenizer,
        model_settings=config["model"],
        training_settings=settings,
        seed=int(settings["final_seed"]),
        epochs=final_epochs,
        device=device,
    )
    logits = _predict_logits(
        final_model,
        tokenizer,
        test_reviews,
        prepared.test_labels,
        objective,
        settings,
        device,
    )
    metrics, predictions, probabilities = _evaluate(
        objective, prepared.test_labels, prepared.test_languages, logits
    )
    model_dir = output_dir / "models" / objective
    save_robbert_artifact(
        final_model,
        tokenizer,
        model_dir,
        model_id=config["model"]["model_id"],
        revision=config["model"]["revision"],
        objective=objective,
        max_length=int(settings["max_length"]),
    )
    del final_model
    return {
        "objective": objective,
        "validation_runs": validation_runs,
        "selected_epoch": final_epochs,
        "final_seed": int(settings["final_seed"]),
        "final_training_history": final_history,
        "test_metrics": metrics,
        "test_predictions": predictions,
        "test_probabilities": probabilities,
        "model_dir": str(model_dir),
        "final_fit_and_test_seconds": time.perf_counter() - started,
    }


def _write_checksums(output_dir: Path) -> None:
    """Write deterministic checksums for every material bundle file."""
    files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(output_dir)}" for path in files]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    config_path: str | Path, output_override: str | Path | None = None
) -> dict[str, Any]:
    """Run both frozen RobBERT candidates and create a portable result bundle."""
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
    candidates = [
        _run_candidate(objective, prepared, tokenizer, config, output_dir, device)
        for objective in config["candidates"]
    ]
    prediction_frame = pd.DataFrame(
        {
            "row_index": range(prepared.heldout_rows),
            "review_sha256": [
                hashlib.sha256(normalize_text(value).encode()).hexdigest()
                for value in prepared.test_reviews
            ],
            "actual": prepared.test_labels,
            "detected_language": prepared.test_languages,
        }
    )
    for candidate in candidates:
        name = candidate["objective"]
        prediction_frame[f"{name}_prediction"] = candidate.pop("test_predictions")
        probabilities = candidate.pop("test_probabilities")
        for index, label in enumerate(LABELS):
            prediction_frame[f"{name}_p_{label.lower()}"] = probabilities[:, index]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    result = {
        "experiment": "robbert-v2-paired-finetuning-v1",
        "evaluation_scope": "train-validation-selection-and-reused-test-evidence",
        "test_used_for_selection": False,
        "production_champion_changed": False,
        "presentation_selected": False,
        "source_commit": _git_commit(),
        "data": {
            "raw_sha256": prepared.raw_sha256,
            "train_normalized_sha256": prepared.train_sha256,
            "test_normalized_sha256": prepared.heldout_sha256,
            "train_rows": prepared.train_rows,
            "test_rows": prepared.heldout_rows,
            "validation_fold": int(config["training"]["validation_fold"]),
        },
        "model": config["model"],
        "training": config["training"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
        "candidates": candidates,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_checksums(output_dir)
    return result


def verify_bundle(bundle_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Verify bundle checksums, frozen hashes, row count, and probability contracts."""
    root = Path(bundle_dir)
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing bundle checksum file: {checksum_path}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"Bundle checksum mismatch: {relative}")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    result = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    if result["data"]["train_normalized_sha256"] != config["expected_train_normalized_sha256"]:
        raise RuntimeError("Bundle training split hash does not match the frozen config")
    if result["data"]["test_normalized_sha256"] != config["expected_test_normalized_sha256"]:
        raise RuntimeError("Bundle test split hash does not match the frozen config")
    predictions = pd.read_csv(root / "test_predictions.csv")
    if len(predictions) != int(result["data"]["test_rows"]):
        raise RuntimeError("Bundle prediction row count does not match its manifest")
    if [candidate["objective"] for candidate in result["candidates"]] != list(OBJECTIVES):
        raise RuntimeError("Bundle does not contain the paired frozen RobBERT candidates")
    return result


def log_existing_bundle(bundle_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Import a verified Colab bundle into local MLflow without retraining."""
    import mlflow
    from mlflow import MlflowClient

    root = Path(bundle_dir)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    result = verify_bundle(root, config_path)
    tracking_uri = str(config["mlflow_tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(config["mlflow_experiment"])
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(
            config["mlflow_experiment"], tags={"purpose": "paired-robbert-finetuning"}
        )
    )
    matches = client.search_runs(
        [experiment_id], filter_string=f"tags.catalog_id = '{config['mlflow_catalog_id']}'"
    )
    if matches:
        result["mlflow_run_id"] = str(matches[0].info.run_id)
        return result
    tags = {
        "catalog_id": config["mlflow_catalog_id"],
        "evaluation.scope": result["evaluation_scope"],
        "test_used_for_selection": "false",
        "production_champion_changed": "false",
        "presentation.selected": "false",
    }
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="robbert-v2-paired", tags=tags
    ) as run:
        for candidate in result["candidates"]:
            prefix = candidate["objective"]
            for key, value in candidate["test_metrics"].items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    mlflow.log_metric(f"{prefix}.test_{key}", float(value))
            mlflow.log_artifacts(str(root / "models" / prefix), artifact_path=f"models/{prefix}")
        mlflow.log_artifacts(str(root), artifact_path="evidence")
        result["mlflow_run_id"] = str(run.info.run_id)
    return result


def main() -> None:
    """Run Colab training, verify a bundle, or import it into local MLflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/robbert_v2.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--verify-bundle")
    parser.add_argument("--log-existing")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.verify_bundle, args.log_existing)) > 1:
        parser.error("Choose only one bundle operation")
    if args.verify_bundle:
        result = verify_bundle(args.verify_bundle, args.config)
    elif args.log_existing:
        result = log_existing_bundle(args.log_existing, args.config)
    else:
        result = run_experiment(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
