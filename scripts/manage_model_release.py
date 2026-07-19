"""Export or verify the tracked production artifact against the MLflow champion."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dutch_sentiment.release import ReleaseManifest, sha256_file, verify_release_files


def write_champion_manifest(
    tracking_uri: str, manifest_path: Path, registry_model: str, registry_alias: str
) -> dict[str, str]:
    """Write a reviewable manifest for the current champion before exporting it."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    version = client.get_model_version_by_alias(registry_model, registry_alias)
    registry_run = client.get_run(str(version.run_id))
    source_run_id = str(registry_run.data.tags.get("source_run_id", ""))
    if not source_run_id:
        raise ValueError("Champion registry run does not identify its source_run_id")
    source_model = Path(client.download_artifacts(source_run_id, "evidence/model.joblib"))
    source_metadata = Path(client.download_artifacts(source_run_id, "evidence/model_metadata.json"))
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    payload = {
        "registry_model": registry_model,
        "registry_alias": registry_alias,
        "registry_version": str(version.version),
        "registry_run_id": str(version.run_id),
        "source_run_id": source_run_id,
        "model_version": str(metadata["model_version"]),
        "model_sha256": sha256_file(source_model),
        "metadata_sha256": sha256_file(source_metadata),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _champion_evidence(tracking_uri: str, manifest: ReleaseManifest) -> tuple[object, Path, Path]:
    """Resolve the champion and download its exact source-run release artifacts."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    version = client.get_model_version_by_alias(manifest.registry_model, manifest.registry_alias)
    if str(version.version) != manifest.registry_version:
        raise ValueError("MLflow champion version differs from the release manifest")
    if str(version.run_id) != manifest.registry_run_id:
        raise ValueError("MLflow registry run differs from the release manifest")
    registry_run = client.get_run(str(version.run_id))
    source_run_id = str(registry_run.data.tags.get("source_run_id", ""))
    if source_run_id != manifest.source_run_id:
        raise ValueError("MLflow champion source run differs from the release manifest")
    model = Path(client.download_artifacts(source_run_id, "evidence/model.joblib"))
    metadata = Path(client.download_artifacts(source_run_id, "evidence/model_metadata.json"))
    return version, model, metadata


def verify_mlflow(
    tracking_uri: str, manifest_path: Path, model_path: Path, metadata_path: Path
) -> dict[str, str]:
    """Verify local release files and the champion's immutable source-run copies."""
    result = verify_release_files(model_path, metadata_path, manifest_path)
    manifest = ReleaseManifest.load(manifest_path)
    _, source_model, source_metadata = _champion_evidence(tracking_uri, manifest)
    if sha256_file(source_model) != result["model_sha256"]:
        raise ValueError("MLflow source model differs from the deployed artifact")
    if sha256_file(source_metadata) != result["metadata_sha256"]:
        raise ValueError("MLflow source metadata differs from the deployed metadata")
    return {**result, "tracking_uri": tracking_uri, "mlflow_source_verified": "true"}


def export_champion(
    tracking_uri: str, manifest_path: Path, model_path: Path, metadata_path: Path
) -> dict[str, str]:
    """Replace deployment files with the exact artifacts behind the champion alias."""
    manifest = ReleaseManifest.load(manifest_path)
    _, source_model, source_metadata = _champion_evidence(tracking_uri, manifest)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model, model_path)
    shutil.copy2(source_metadata, metadata_path)
    return verify_mlflow(tracking_uri, manifest_path, model_path, metadata_path)


def parse_args() -> argparse.Namespace:
    """Parse release paths and whether MLflow is required."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "manifest", "export"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/model_metadata.json"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/model_release.json"))
    parser.add_argument("--registry-model", default="sentiment-production")
    parser.add_argument("--registry-alias", default="champion")
    parser.add_argument(
        "--require-mlflow",
        action="store_true",
        help="Also compare with the champion source run; required for local release promotion.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a read-only verification or an explicit champion export."""
    args = parse_args()
    if args.command == "manifest":
        result = write_champion_manifest(
            args.tracking_uri, args.manifest, args.registry_model, args.registry_alias
        )
    elif args.command == "export":
        result = export_champion(args.tracking_uri, args.manifest, args.model, args.metadata)
    elif args.require_mlflow:
        result = verify_mlflow(args.tracking_uri, args.manifest, args.model, args.metadata)
    else:
        result = verify_release_files(args.model, args.metadata, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
