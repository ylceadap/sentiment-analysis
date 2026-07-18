"""Compare bounded CPU-friendly linear candidates without touching held-out labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from .config import load_config
from .data import annotate_review_languages, load_dataset, make_holdout_split
from .language import DutchLanguageDetector
from .metrics import classification_metrics
from .model import ModelSpec, build_pipeline

LOGGER = logging.getLogger(__name__)
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)


@dataclass(frozen=True)
class LinearCandidate:
    name: str
    classifier_kind: str
    regularization_c: float


def _hash_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def candidate_specs(config: dict[str, Any]) -> list[LinearCandidate]:
    """Build the intentionally small, predeclared candidate list."""
    logistic = [
        LinearCandidate(f"logreg_c{value:g}", "logistic_regression", float(value))
        for value in config["logistic_c_values"]
    ]
    linear_svc = [
        LinearCandidate(f"linear_svc_c{value:g}", "linear_svc", float(value))
        for value in config["linear_svc_c_values"]
    ]
    return logistic + linear_svc


def build_candidate_pipeline(
    candidate: LinearCandidate, model_config: dict[str, Any], seed: int
) -> Any:
    """Reuse the frozen feature pipeline and swap only classifier/C."""
    pipeline = build_pipeline(
        ModelSpec("linear_experiment", "combined", class_weight="balanced"),
        {**model_config, "random_seed": seed},
    )
    if candidate.classifier_kind == "logistic_regression":
        pipeline.set_params(classifier__C=candidate.regularization_c)
    elif candidate.classifier_kind == "linear_svc":
        pipeline.set_params(
            classifier=LinearSVC(
                C=candidate.regularization_c,
                class_weight="balanced",
                random_state=seed,
                max_iter=int(model_config.get("max_iter", 1500)),
            )
        )
    else:
        raise ValueError(f"Unsupported classifier kind: {candidate.classifier_kind}")
    return pipeline


def _language_metrics(
    languages: list[str], labels: list[str], predictions: list[str]
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for language in sorted(set(languages)):
        indices = [index for index, value in enumerate(languages) if value == language]
        metrics[language] = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
    return metrics


def evaluate_candidate(
    candidate: LinearCandidate,
    reviews: list[str],
    labels: list[str],
    languages: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Fit once per fold and retain only out-of-fold training predictions."""
    oof_predictions = np.empty(len(labels), dtype=object)
    fold_metrics: list[dict[str, Any]] = []
    fold_fit_seconds: list[float] = []
    review_array = np.asarray(reviews, dtype=object)
    label_array = np.asarray(labels, dtype=object)
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        pipeline = build_candidate_pipeline(candidate, model_config, seed)
        started = perf_counter()
        pipeline.fit(review_array[train_indices].tolist(), label_array[train_indices].tolist())
        fold_fit_seconds.append(perf_counter() - started)
        predictions = pipeline.predict(review_array[validation_indices].tolist()).tolist()
        oof_predictions[validation_indices] = predictions
        metrics = classification_metrics(label_array[validation_indices].tolist(), predictions)
        metrics["fold"] = fold_number
        fold_metrics.append(metrics)

    prediction_list = [str(value) for value in oof_predictions.tolist()]
    overall = classification_metrics(labels, prediction_list)
    by_language = _language_metrics(languages, labels, prediction_list)
    row: dict[str, Any] = {
        **asdict(candidate),
        "cv_folds": len(folds),
        "cv_fit_seconds": sum(fold_fit_seconds),
        "mean_fold_fit_seconds": float(np.mean(fold_fit_seconds)),
        "oof_macro_f1": overall["macro_f1"],
        "oof_negative_f1": overall["per_class"]["Negative"]["f1-score"],
        "oof_dutch_macro_f1": by_language["dutch"]["macro_f1"],
        "oof_english_macro_f1": by_language["english"]["macro_f1"],
    }
    for metric in SUMMARY_METRICS:
        values = np.asarray([fold[metric] for fold in fold_metrics], dtype=float)
        row[f"cv_{metric}_mean"] = float(values.mean())
        row[f"cv_{metric}_std"] = float(values.std())
    return row


def decide_promotion(results: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Apply predeclared gates relative to the exact Logistic C=1 baseline."""
    baseline = results.loc[results["name"].eq("logreg_c1")].iloc[0]
    best = results.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
    gates = config["promotion"]
    checks = {
        "macro_f1_improvement": float(best["cv_macro_f1_mean"] - baseline["cv_macro_f1_mean"])
        >= float(gates["minimum_macro_f1_improvement"]),
        "dutch_macro_f1_guardrail": float(best["oof_dutch_macro_f1"])
        >= float(baseline["oof_dutch_macro_f1"]) - float(gates["maximum_dutch_macro_f1_drop"]),
        "negative_f1_guardrail": float(best["oof_negative_f1"])
        >= float(baseline["oof_negative_f1"]) - float(gates["maximum_negative_f1_drop"]),
    }
    promote = best["name"] != baseline["name"] and all(checks.values())
    return {
        "baseline": str(baseline["name"]),
        "best_candidate": str(best["name"]),
        "macro_f1_improvement": float(best["cv_macro_f1_mean"] - baseline["cv_macro_f1_mean"]),
        "checks": checks,
        "promote": bool(promote),
    }


def _write_report(
    path: str | Path,
    results: pd.DataFrame,
    decision: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    columns = [
        "name",
        "classifier_kind",
        "regularization_c",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "oof_negative_f1",
        "oof_dutch_macro_f1",
        "oof_english_macro_f1",
        "cv_fit_seconds",
    ]
    table = results[columns].round(4).to_markdown(index=False)
    recommendation = (
        f"Promote `{decision['best_candidate']}` to a separately approved held-out evaluation."
        if decision["promote"]
        else "Do not replace `submission-v1`; no candidate passed all predeclared gates."
    )
    report = f"""# Linear Model Experiment

## Scope and leakage control

- Branch-only experiment; `main` and `submission-v1` remain unchanged.
- Git branch: `{evidence["git_branch"]}`; commit: `{evidence["git_commit"]}`.
- Working tree dirty when experiment started: **{str(evidence["git_dirty"]).lower()}**.
- Source rows: {evidence["raw_rows"]}; training rows used: {evidence["training_rows"]}.
- Frozen training hash: `{evidence["train_normalized_sha256"]}`.
- Frozen held-out hash: `{evidence["test_normalized_sha256"]}`.
- Held-out labels evaluated: **no**.
- Candidate selection uses the same language×label-stratified {evidence["cv_folds"]}-fold indices.
- Frozen Logistic C=1 CV macro-F1: {evidence["frozen_baseline_cv_macro_f1"]:.12f}.
- Reproduced Logistic C=1 CV macro-F1: {evidence["reproduced_baseline_cv_macro_f1"]:.12f}.

## Results

{table}

## Predeclared decision

```json
{json.dumps(decision, indent=2)}
```

**Recommendation:** {recommendation}

The English slice is descriptive and small. LinearSVC candidates do not provide native probabilities; if one ever passes the CV gates, probability calibration and its latency must be evaluated before API promotion.
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    training_config = load_config(config["training_config"])
    seed = int(training_config["random_seed"])
    raw = load_dataset(training_config["data_path"])
    detector = DutchLanguageDetector(**training_config["language"])
    annotated, _ = annotate_review_languages(raw, detector)
    split = make_holdout_split(
        annotated,
        test_size=float(training_config["test_size"]),
        random_seed=seed,
        stratify_columns=("detected_language", "Label"),
    )
    frozen = json.loads(Path("artifacts/split_metadata.json").read_text(encoding="utf-8"))
    train_hash = frozen["train_normalized_sha256"]
    test_hash = frozen["test_normalized_sha256"]
    actual_train_hash = _hash_values(split.train["normalized_review"].tolist())
    actual_test_hash = _hash_values(split.test["normalized_review"].tolist())
    if actual_train_hash != train_hash or actual_test_hash != test_hash:
        raise RuntimeError("Experiment split does not match frozen submission hashes")

    reviews = split.train["Reviews"].astype(str).tolist()
    labels = split.train["Label"].astype(str).tolist()
    languages = split.train["detected_language"].astype(str).tolist()
    strata = split.train[["detected_language", "Label"]].astype(str).agg("::".join, axis=1)
    cv = StratifiedKFold(n_splits=int(training_config["cv_folds"]), shuffle=True, random_state=seed)
    folds = list(cv.split(reviews, strata))
    rows = []
    for candidate in candidate_specs(config):
        LOGGER.info("Evaluating %s without held-out access", candidate.name)
        rows.append(
            evaluate_candidate(
                candidate,
                reviews,
                labels,
                languages,
                folds,
                training_config["model"],
                seed,
            )
        )
    results = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    frozen_comparison = pd.read_csv("artifacts/experiment_comparison.csv")
    frozen_baseline_score = float(
        frozen_comparison.loc[
            frozen_comparison["name"].eq("combined_balanced_ratings"),
            "cv_macro_f1_mean",
        ].iloc[0]
    )
    reproduced_baseline_score = float(
        results.loc[results["name"].eq("logreg_c1"), "cv_macro_f1_mean"].iloc[0]
    )
    if not np.isclose(frozen_baseline_score, reproduced_baseline_score, atol=1e-12):
        raise RuntimeError("Logistic C=1 did not reproduce the frozen CV baseline")
    decision = decide_promotion(results, config)
    output_csv = Path(config["output_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    evidence = {
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "raw_rows": len(raw),
        "training_rows": len(split.train),
        "cv_folds": len(folds),
        "train_normalized_sha256": actual_train_hash,
        "test_normalized_sha256": actual_test_hash,
        "heldout_evaluated": False,
        "frozen_baseline_cv_macro_f1": frozen_baseline_score,
        "reproduced_baseline_cv_macro_f1": reproduced_baseline_score,
    }
    _write_report(config["output_report"], results, decision, evidence)
    result = {"decision": decision, "evidence": evidence}
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/linear_experiment.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(args.config)


if __name__ == "__main__":
    main()
