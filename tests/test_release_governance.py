import json
from pathlib import Path

import pytest

from dutch_sentiment.models.classical import ModelSpec, SentimentModel, build_pipeline
from dutch_sentiment.release import sha256_file, verify_release_files


def _release_model(path: Path) -> SentimentModel:
    """Create a tiny fitted release artifact for contract tests."""
    reviews = [
        "prachtig en goed",
        "heel mooi gespeeld",
        "gewoon gemiddeld",
        "redelijk maar matig",
        "verschrikkelijk slecht",
        "saai en waardeloos",
    ]
    labels = ["Positive", "Positive", "Average", "Average", "Negative", "Negative"]
    config = {"min_df": 1, "max_df": 1.0, "max_iter": 100, "random_seed": 42}
    model = SentimentModel(
        build_pipeline(ModelSpec("release-test", "word", "balanced"), config), "release-v1"
    ).fit(reviews, labels)
    model.save(path)
    return model


def test_release_verifies_artifact_metadata_and_manifest(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    _release_model(model_path)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "release-v1",
                "model_sha256": sha256_file(model_path),
                "mlflow_run_id": "source-run",
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(
        json.dumps(
            {
                "registry_model": "sentiment-production",
                "registry_alias": "champion",
                "registry_version": "1",
                "registry_run_id": "registry-run",
                "source_run_id": "source-run",
                "model_version": "release-v1",
                "model_sha256": sha256_file(model_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        ),
        encoding="utf-8",
    )
    result = verify_release_files(model_path, metadata_path, manifest_path)
    assert result["registry_alias"] == "champion"
    assert result["model_sha256"] == sha256_file(model_path)


def test_release_rejects_tampered_model(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"tampered")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"model_version": "v1", "model_sha256": "a" * 64, "mlflow_run_id": "run"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(
        json.dumps(
            {
                "registry_model": "production",
                "registry_alias": "champion",
                "registry_version": "1",
                "registry_run_id": "registry",
                "source_run_id": "run",
                "model_version": "v1",
                "model_sha256": "a" * 64,
                "metadata_sha256": sha256_file(metadata_path),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Model SHA-256 differs"):
        verify_release_files(model_path, metadata_path, manifest_path, load_model=False)
