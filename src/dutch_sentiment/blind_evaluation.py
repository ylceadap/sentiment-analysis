"""Evaluate frozen deployable candidates on a genuinely unseen labeled dataset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import load_config
from .constants import LABELS
from .data import load_dataset, sha256_file
from .metrics import classification_metrics
from .models.classical import SentimentModel
from .text import normalize_text


def validate_blind_dataset(
    blind_path: str | Path,
    *,
    expected_sha256: str,
    source_data_path: str | Path,
    minimum_rows_per_label: int,
) -> tuple[Any, str]:
    """Reject unsealed, reused, overlapping, or underpowered blind datasets."""
    if len(expected_sha256) != 64 or set(expected_sha256) == {"0"}:
        raise ValueError("Set a real expected_sha256 before opening the blind labels")
    actual_sha256 = sha256_file(blind_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Blind dataset SHA-256 does not match its sealed configuration")
    if actual_sha256 == sha256_file(source_data_path):
        raise ValueError("The original assignment CSV cannot be reused as a blind test")

    blind = load_dataset(blind_path)
    source = load_dataset(source_data_path)
    blind_normalized = blind["Reviews"].map(normalize_text)
    source_normalized = set(source["Reviews"].map(normalize_text))
    overlap = sorted(set(blind_normalized) & source_normalized)
    if overlap:
        raise ValueError(f"Blind dataset overlaps the known assignment data in {len(overlap)} rows")
    if blind_normalized.duplicated().any():
        raise ValueError("Blind dataset contains normalized duplicate reviews")
    counts = blind["Label"].value_counts().to_dict()
    insufficient = {
        label: int(counts.get(label, 0))
        for label in LABELS
        if int(counts.get(label, 0)) < minimum_rows_per_label
    }
    if insufficient:
        raise ValueError(f"Blind dataset has insufficient per-label support: {insufficient}")
    return blind, actual_sha256


def evaluate_candidates(frame: Any, candidates: dict[str, str]) -> dict[str, Any]:
    """Evaluate only frozen, locally loadable candidate artifacts."""
    reviews = frame["Reviews"].astype(str).tolist()
    labels = frame["Label"].astype(str).tolist()
    results: dict[str, Any] = {}
    for name, model_path in candidates.items():
        model = SentimentModel.load(model_path)
        predictions = model.predict(reviews)
        probabilities = model.predict_proba(reviews)
        results[name] = {
            "model_path": str(model_path),
            "model_version": model.version,
            "metrics": classification_metrics(labels, predictions, probabilities),
        }
    return results


def run(config_path: str | Path, *, confirmed_unseen: bool) -> dict[str, Any]:
    """Validate the sealed dataset and write one immutable comparison result."""
    if not confirmed_unseen:
        raise ValueError("Pass --confirm-unseen only after labels and model choices are frozen")
    config = load_config(config_path)
    frame, digest = validate_blind_dataset(
        config["dataset_path"],
        expected_sha256=str(config["expected_sha256"]),
        source_data_path=config.get("known_source_data", "Python_Engineer_Challenge_2.csv"),
        minimum_rows_per_label=int(config.get("minimum_rows_per_label", 30)),
    )
    candidates = {str(key): str(value) for key, value in config["candidates"].items()}
    if len(candidates) < 2:
        raise ValueError("Blind evaluation requires production and at least one challenger")
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_scope": "new-blind-test",
        "dataset_sha256": digest,
        "rows": len(frame),
        "label_counts": frame["Label"].value_counts().to_dict(),
        "candidates": evaluate_candidates(frame, candidates),
    }
    output = Path(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    """Parse the sealed blind-evaluation configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/blind_evaluation.yaml")
    parser.add_argument("--confirm-unseen", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute the one-time blind comparison after explicit confirmation."""
    args = parse_args()
    payload = run(args.config, confirmed_unseen=args.confirm_unseen)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
