"""Verify the immutable production-model release contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models.classical import SentimentModel


def sha256_file(path: str | Path) -> str:
    """Return a file digest without modifying the release artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and reject other JSON top-level values."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


@dataclass(frozen=True)
class ReleaseManifest:
    """Bind the served artifact to its MLflow champion and training evidence."""

    registry_model: str
    registry_alias: str
    registry_version: str
    registry_run_id: str
    source_run_id: str
    model_version: str
    model_sha256: str
    metadata_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> ReleaseManifest:
        """Load and validate the tracked release manifest."""
        payload = _load_object(path)
        missing = sorted(set(cls.__dataclass_fields__) - set(payload))
        if missing:
            raise ValueError(f"Release manifest is missing fields: {missing}")
        return cls(**{field: str(payload[field]) for field in cls.__dataclass_fields__})


def verify_release_files(
    model_path: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    *,
    load_model: bool = True,
) -> dict[str, str]:
    """Verify model, metadata, and manifest identity before packaging or serving."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    manifest = ReleaseManifest.load(manifest_path)
    metadata = _load_object(metadata_path)
    model_hash = sha256_file(model_path)
    metadata_hash = sha256_file(metadata_path)

    expected_model_hashes = {
        model_hash,
        str(metadata.get("model_sha256", "")),
        manifest.model_sha256,
    }
    if len(expected_model_hashes) != 1:
        raise ValueError("Model SHA-256 differs across artifact, metadata, and release manifest")
    if metadata_hash != manifest.metadata_sha256:
        raise ValueError("Metadata SHA-256 differs from the release manifest")
    if str(metadata.get("model_version")) != manifest.model_version:
        raise ValueError("Model version differs between metadata and release manifest")
    if str(metadata.get("mlflow_run_id")) != manifest.source_run_id:
        raise ValueError("Training run differs between metadata and release manifest")

    if load_model:
        model = SentimentModel.load(model_path)
        if model.version != manifest.model_version:
            raise ValueError("Serialized model version differs from the release manifest")
        inference = model.infer("Deze film was verrassend goed.")
        if abs(sum(inference.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("Production model probabilities do not sum to one")

    return {
        "registry_model": manifest.registry_model,
        "registry_alias": manifest.registry_alias,
        "registry_version": manifest.registry_version,
        "model_version": manifest.model_version,
        "model_sha256": model_hash,
        "metadata_sha256": metadata_hash,
        "source_run_id": manifest.source_run_id,
    }
