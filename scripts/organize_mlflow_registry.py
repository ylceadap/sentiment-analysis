"""Apply the repository's model-governance policy to the local MLflow store."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow import MlflowClient


@dataclass(frozen=True)
class RegisteredModelPolicy:
    """Describe how one loadable registry artifact may be used."""

    alias: str
    description: str
    tags: dict[str, str]
    source_catalog_id: str | None = None
    source_artifact_path: str = "evidence"


@dataclass(frozen=True)
class ExperimentEvidence:
    """Describe branch evidence that is not necessarily a deployable model artifact."""

    catalog_id: str
    branch: str
    run_name: str
    evaluation_scope: str
    promotion_status: str
    files: tuple[str, ...]
    metrics: dict[str, float]
    notes: str


COMMON_LOCAL_TAGS = {
    "artifact.kind": "local-mlflow-model",
    "data.raw_sha256": "2788b987e2c9fa4fd6459a1798a6e7d1dd63ddb5618c10907712d39042cc4be2",
    "data.train_split_sha256": "aa986ef1d8f35ebf232a015fd0e61d3affb1efbe4a589ec4f8653ae01e8ab7c9",
    "data.heldout_split_sha256": "b76afd73eaeed79bf61903ab8475a08c4565cf523a55e8ae34448d2330e00cbb",
    "license.use": "repository-controlled",
}


MODEL_POLICIES = {
    "sentiment-production": RegisteredModelPolicy(
        alias="champion",
        description=(
            "Official served model: shared Dutch-English word+character TF-IDF with balanced "
            "multinomial Logistic Regression. This is the only production champion."
        ),
        tags={
            **COMMON_LOCAL_TAGS,
            "governance.tier": "production",
            "governance.status": "active",
            "deployment.eligible": "true",
            "evaluation.scope": "frozen-heldout",
            "source.branch": "main",
        },
    ),
    "sentiment-dummy-prior": RegisteredModelPolicy(
        alias="baseline",
        description="Dummy-prior benchmark retained as the minimum performance baseline.",
        tags={
            **COMMON_LOCAL_TAGS,
            "governance.tier": "baseline",
            "governance.status": "frozen",
            "deployment.eligible": "false",
            "evaluation.scope": "cross-validation",
            "source.branch": "main",
        },
    ),
    "sentiment-word-logreg": RegisteredModelPolicy(
        alias="benchmark",
        description="Word TF-IDF Logistic Regression benchmark; not selected for serving.",
        tags={
            **COMMON_LOCAL_TAGS,
            "governance.tier": "benchmark",
            "governance.status": "frozen",
            "deployment.eligible": "false",
            "evaluation.scope": "cross-validation",
            "source.branch": "main",
        },
    ),
    "sentiment-char-logreg": RegisteredModelPolicy(
        alias="benchmark",
        description="Character TF-IDF Logistic Regression benchmark; not selected for serving.",
        tags={
            **COMMON_LOCAL_TAGS,
            "governance.tier": "benchmark",
            "governance.status": "frozen",
            "deployment.eligible": "false",
            "evaluation.scope": "cross-validation",
            "source.branch": "main",
        },
    ),
    "sentiment-combined-logreg": RegisteredModelPolicy(
        alias="benchmark",
        description="Unweighted combined word+character TF-IDF benchmark.",
        tags={
            **COMMON_LOCAL_TAGS,
            "governance.tier": "benchmark",
            "governance.status": "frozen",
            "deployment.eligible": "false",
            "evaluation.scope": "cross-validation",
            "source.branch": "main",
        },
    ),
    "sentiment-combined-balanced": RegisteredModelPolicy(
        alias="benchmark",
        description="Balanced combined TF-IDF candidate from which the production artifact was fitted.",
        tags={
            **COMMON_LOCAL_TAGS,
            "governance.tier": "benchmark",
            "governance.status": "selected-training-candidate",
            "deployment.eligible": "false",
            "evaluation.scope": "cross-validation",
            "source.branch": "main",
        },
    ),
    "sentiment-combined-balanced-masked": RegisteredModelPolicy(
        alias="benchmark",
        description="Balanced combined TF-IDF rating-masking ablation; not selected.",
        tags={
            **COMMON_LOCAL_TAGS,
            "governance.tier": "benchmark",
            "governance.status": "frozen",
            "deployment.eligible": "false",
            "evaluation.scope": "cross-validation",
            "source.branch": "main",
        },
    ),
    "sentiment-linear-svc": RegisteredModelPolicy(
        alias="frozen-challenger",
        description="LinearSVC challenger; retained for comparison and not promoted.",
        tags={
            **COMMON_LOCAL_TAGS,
            "governance.tier": "challenger",
            "governance.status": "not-promoted",
            "deployment.eligible": "false",
            "evaluation.scope": "training-oof",
            "source.branch": "experiment/linear-models",
        },
    ),
    "sentiment-frozen-robbert-embeddings": RegisteredModelPolicy(
        alias="research-only",
        description=(
            "Frozen RobBERT embedding research model. The encoder is revision-pinned but not "
            "bundled as a self-contained production artifact."
        ),
        tags={
            **COMMON_LOCAL_TAGS,
            "governance.tier": "research",
            "governance.status": "not-promoted",
            "deployment.eligible": "false",
            "evaluation.scope": "training-oof",
            "source.branch": "experiment/transformer-embeddings",
            "artifact.self_contained": "false",
        },
    ),
    "sentiment-deepseek-v4-flash-24shot": RegisteredModelPolicy(
        alias="external-advisor",
        description=(
            "External DeepSeek API prompt wrapper. This entry contains no provider model weights "
            "and is an advisor, not a locally deployable model."
        ),
        tags={
            "artifact.kind": "remote-api-wrapper",
            "governance.tier": "external-advisor",
            "governance.status": "architecture-review-required",
            "deployment.eligible": "conditional",
            "evaluation.scope": "reused-heldout",
            "source.branch": "experiment/llm",
            "license.use": "provider-terms",
            "privacy.external_processing": "true",
            "weights.stored": "false",
        },
    ),
    "sentiment-jina-v3-logreg": RegisteredModelPolicy(
        alias="research-only",
        description=(
            "Frozen Jina v3 classification embeddings with Logistic Regression. Registered as "
            "research evidence only: no self-contained serving artifact is stored, no blind test "
            "was run, and Jina v3 is licensed for non-commercial use."
        ),
        tags={
            "artifact.kind": "evidence-only-mlflow-entry",
            "governance.tier": "research",
            "governance.status": "research-only",
            "deployment.eligible": "false",
            "evaluation.scope": "training-oof",
            "source.branch": "experiment/jina-embeddings",
            "license.use": "cc-by-nc-4.0",
            "blind_test.completed": "false",
            "artifact.self_contained": "false",
            "weights.stored": "false",
            "encoder.model": "jinaai/jina-embeddings-v3",
        },
        source_catalog_id="jina-embeddings-v1",
    ),
    "sentiment-jina-v3-ordinal-logistic": RegisteredModelPolicy(
        alias="research-only",
        description=(
            "Frozen Jina v3 classification embeddings with a two-boundary ordinal Logistic "
            "Regression head. Registered as the strongest local OOF research candidate, not a "
            "production model: no blind test was run, no self-contained serving artifact is "
            "stored, and Jina v3 is licensed for non-commercial use."
        ),
        tags={
            "artifact.kind": "evidence-only-mlflow-entry",
            "governance.tier": "research",
            "governance.status": "research-only",
            "deployment.eligible": "false",
            "evaluation.scope": "training-oof",
            "source.branch": "experiment/jina-ordinal-logistic",
            "license.use": "cc-by-nc-4.0",
            "blind_test.completed": "false",
            "artifact.self_contained": "false",
            "weights.stored": "false",
            "encoder.model": "jinaai/jina-embeddings-v3",
        },
        source_catalog_id="jina-ordinal-logistic-v1",
    ),
    "sentiment-tfidf-ordinal-logistic": RegisteredModelPolicy(
        alias="frozen-challenger",
        description=(
            "Branch artifact for the TF-IDF two-boundary ordinal Logistic Regression challenger. "
            "It is locally deployable from its experiment branch, but it is not the production "
            "champion because the comparison reused the existing held-out rows and still needs a "
            "new blind benchmark."
        ),
        tags={
            **COMMON_LOCAL_TAGS,
            "artifact.kind": "branch-model-artifact",
            "governance.tier": "challenger",
            "governance.status": "needs-new-blind-test",
            "deployment.eligible": "false",
            "evaluation.scope": "reused-heldout",
            "source.branch": "experiment/ordinal-logistic",
            "blind_test.completed": "false",
            "artifact.self_contained": "true",
            "weights.stored": "true",
        },
        source_catalog_id="ordinal-logistic-v1",
    ),
}


EVIDENCE = (
    ExperimentEvidence(
        "linear-models-v1",
        "experiment/linear-models",
        "evidence-linear-models",
        "training-oof",
        "not-promoted",
        (
            "configs/linear_experiment.yaml",
            "artifacts/linear_model_experiment.csv",
            "reports/linear_model_experiment.md",
        ),
        {"best_oof_macro_f1": 0.6536},
        "C sweep and LinearSVC comparison; predefined improvement gate did not pass.",
    ),
    ExperimentEvidence(
        "negative-imbalance-v1",
        "experiment/negative-imbalance",
        "evidence-negative-imbalance",
        "training-oof-and-reused-heldout",
        "not-promoted",
        (
            "configs/imbalance_experiment.yaml",
            "artifacts/imbalance_oof_decision.json",
            "artifacts/imbalance_heldout_comparison.json",
            "artifacts/imbalance_heldout_lock.json",
            "reports/negative_imbalance_experiment.md",
        ),
        {
            "heldout_macro_f1": 0.6396,
            "heldout_negative_precision": 0.5909,
            "heldout_negative_recall": 0.65,
        },
        "Improved Negative recall but missed the predefined precision floor.",
    ),
    ExperimentEvidence(
        "transformer-embeddings-v1",
        "experiment/transformer-embeddings",
        "evidence-transformer-embeddings",
        "training-oof",
        "not-promoted",
        (
            "configs/embedding_experiment.yaml",
            "artifacts/embedding_experiment_decision.json",
            "artifacts/embedding_experiment_results.csv",
            "reports/embedding_experiment.md",
        ),
        {"best_oof_macro_f1": 0.5165},
        "Frozen MiniLM/RobBERT candidates failed the promotion gates.",
    ),
    ExperimentEvidence(
        "jina-embeddings-v1",
        "experiment/jina-embeddings",
        "evidence-jina-embeddings",
        "training-oof",
        "research-only",
        (
            "configs/embedding_experiment.yaml",
            "artifacts/jina_embedding_experiment_decision.json",
            "artifacts/jina_embedding_experiment_results.csv",
            "reports/jina_embedding_experiment.md",
            "reports/jina_embedding_validation.md",
        ),
        {
            "best_oof_macro_f1": 0.7108,
            "best_oof_negative_precision": 0.6196,
            "best_oof_negative_recall": 0.8958,
        },
        "Jina v3 passed OOF gates but has no new blind test and uses a non-commercial license.",
    ),
    ExperimentEvidence(
        "llm-deepseek-v1",
        "experiment/llm",
        "evidence-deepseek-v4-flash",
        "reused-heldout",
        "architecture-review-required",
        (
            "configs/llm_experiment.yaml",
            "artifacts/model_comparison_validation.json",
            "reports/llm_experiment.md",
        ),
        {
            "heldout_macro_f1": 0.7506,
            "heldout_accuracy": 0.7208,
            "heldout_negative_precision": 0.7746,
            "heldout_negative_recall": 0.9167,
        },
        "External API experiment; the provider weights are not stored locally.",
    ),
    ExperimentEvidence(
        "ordinal-regression-v1",
        "experiment/ordinal-regression",
        "evidence-ordinal-regression",
        "training-oof",
        "not-promoted",
        (
            "artifacts/ordinal/ordinal_experiment.csv",
            "artifacts/ordinal/ordinal_experiment.json",
        ),
        {"selected_oof_macro_f1": 0.648456},
        "Cost-sensitive ordinal decisions did not beat the multiclass baseline gates.",
    ),
    ExperimentEvidence(
        "ordinal-logistic-v1",
        "experiment/ordinal-logistic",
        "evidence-ordinal-logistic",
        "training-oof-and-reused-heldout",
        "deployable-challenger-needs-new-blind-test",
        (
            "artifacts/ordinal_logistic/ordinal_logistic_experiment.csv",
            "artifacts/ordinal_logistic/ordinal_logistic_experiment.json",
            "artifacts/ordinal_logistic/ordinal_logistic_held_out_evaluation.json",
            "artifacts/model.joblib",
            "artifacts/model_metadata.json",
        ),
        {
            "heldout_macro_f1": 0.6406017383,
            "heldout_balanced_accuracy": 0.6403703704,
            "heldout_negative_precision": 0.6379310345,
            "heldout_negative_recall": 0.6166666667,
        },
        "Loadable branch artifact; promotion requires a genuinely new blind benchmark.",
    ),
    ExperimentEvidence(
        "jina-ordinal-logistic-v1",
        "experiment/jina-ordinal-logistic",
        "evidence-jina-ordinal-logistic",
        "training-oof",
        "research-only",
        (
            "configs/jina_ordinal_logistic.yaml",
            "artifacts/jina_ordinal_logistic/jina_ordinal_logistic_experiment.csv",
            "artifacts/jina_ordinal_logistic/jina_ordinal_logistic_experiment.json",
            "reports/jina_ordinal_logistic_experiment.md",
        ),
        {
            "best_oof_macro_f1": 0.7299,
            "best_oof_negative_precision": 0.8089,
            "best_oof_negative_recall": 0.7583,
        },
        "Strongest local OOF evidence; no blind test and Jina v3 is non-commercial research use.",
    ),
)


def _git_bytes(branch: str, path: str) -> bytes | None:
    """Read a tracked file from a branch without changing the working tree."""
    result = subprocess.run(["git", "show", f"{branch}:{path}"], capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def _git_commit(branch: str) -> str:
    """Return the immutable commit recorded for an evidence branch."""
    result = subprocess.run(
        ["git", "rev-parse", branch], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _evidence_runs_by_catalog_id(
    client: MlflowClient, experiment_name: str
) -> dict[str, mlflow.entities.Run]:
    """Return catalog evidence runs keyed by their stable catalog id."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {}
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    return {str(run.data.tags["catalog_id"]): run for run in runs if "catalog_id" in run.data.tags}


def _ensure_registered_model(
    client: MlflowClient,
    name: str,
    policy: RegisteredModelPolicy,
    evidence_runs: dict[str, mlflow.entities.Run],
) -> None:
    """Create an evidence-backed registry entry when the policy allows it."""
    try:
        client.get_registered_model(name)
        return
    except Exception:
        if policy.source_catalog_id is None:
            raise
    evidence = evidence_runs.get(str(policy.source_catalog_id))
    if evidence is None:
        raise RuntimeError(f"Cannot create {name}: missing evidence run {policy.source_catalog_id}")
    source = f"{evidence.info.artifact_uri.rstrip('/')}/{policy.source_artifact_path}"
    client.create_registered_model(name)
    client.create_model_version(
        name=name,
        source=source,
        run_id=evidence.info.run_id,
        description=policy.description,
    )


def apply_model_policies(
    client: MlflowClient,
    evidence_runs: dict[str, mlflow.entities.Run] | None = None,
) -> None:
    """Replace ambiguous aliases and attach governance metadata to registered models."""
    evidence_runs = evidence_runs or {}
    for name, policy in MODEL_POLICIES.items():
        _ensure_registered_model(client, name, policy, evidence_runs)

    existing = {model.name: model for model in client.search_registered_models()}
    missing = sorted(set(MODEL_POLICIES) - set(existing))
    if missing:
        raise RuntimeError(f"Expected registered models are missing: {missing}")

    for name, policy in MODEL_POLICIES.items():
        model = existing[name]
        for alias in model.aliases:
            client.delete_registered_model_alias(name, alias)
        client.update_registered_model(name, description=policy.description)
        client.update_model_version(name, "1", description=policy.description)
        for key, value in policy.tags.items():
            client.set_registered_model_tag(name, key, value)
            client.set_model_version_tag(name, "1", key, value)
        client.set_registered_model_alias(name, policy.alias, "1")


def archive_experiment_evidence(client: MlflowClient, experiment_name: str) -> None:
    """Create one idempotent evidence run for each completed Git experiment branch."""
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(experiment_name, tags={"purpose": "evidence-only-catalog"})
    )
    for item in EVIDENCE:
        matches = client.search_runs(
            [experiment_id], filter_string=f"tags.catalog_id = '{item.catalog_id}'"
        )
        if matches:
            continue
        commit = _git_commit(item.branch)
        tags = {
            "catalog_id": item.catalog_id,
            "evidence_only": "true",
            "model_registry_artifact": "false",
            "source_branch": item.branch,
            "source_commit": commit,
            "evaluation_scope": item.evaluation_scope,
            "promotion_status": item.promotion_status,
            "notes": item.notes,
        }
        with mlflow.start_run(experiment_id=experiment_id, run_name=item.run_name, tags=tags):
            mlflow.log_metrics(item.metrics)
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                written = 0
                for path in item.files:
                    payload = _git_bytes(item.branch, path)
                    if payload is None:
                        continue
                    output = root / path
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(payload)
                    written += 1
                if written == 0:
                    raise RuntimeError(f"No evidence files found for {item.catalog_id}")
                mlflow.log_artifacts(root, artifact_path="evidence")


def parse_args() -> argparse.Namespace:
    """Parse the local tracking URI and evidence experiment name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="dutch-sentiment-research-evidence")
    return parser.parse_args()


def main() -> None:
    """Apply registry policy and archive branch evidence."""
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)
    archive_experiment_evidence(client, args.experiment)
    evidence_runs = _evidence_runs_by_catalog_id(client, args.experiment)
    apply_model_policies(client, evidence_runs)
    print(
        f"Organized {len(MODEL_POLICIES)} registered models and "
        f"cataloged {len(EVIDENCE)} experiment evidence records."
    )


if __name__ == "__main__":
    main()
