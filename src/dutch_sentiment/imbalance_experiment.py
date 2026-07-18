"""Leakage-safe Negative-class weighting, threshold, and oversampling experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.model_selection import StratifiedKFold

from .config import load_config
from .constants import LABELS
from .data import annotate_review_languages, load_dataset, make_holdout_split
from .language import DutchLanguageDetector
from .metrics import classification_metrics
from .model import ModelSpec, SentimentModel, build_pipeline

LOGGER = logging.getLogger(__name__)
AGGREGATE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
CLASS_METRICS = ("precision", "recall", "f1-score")


@dataclass(frozen=True)
class ImbalanceCandidate:
    name: str
    strategy: str
    class_weight: str | dict[str, float] | None = None
    negative_multiplier: int | None = None
    complexity_rank: int = 0


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _hash_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def candidate_specs(config: dict[str, Any]) -> list[ImbalanceCandidate]:
    candidates = []
    for index, item in enumerate(config["class_weight_candidates"]):
        value = item["value"]
        candidates.append(
            ImbalanceCandidate(
                name=str(item["name"]),
                strategy="class_weight",
                class_weight=value if isinstance(value, str) else dict(value),
                complexity_rank=index,
            )
        )
    for index, item in enumerate(config["oversampling_candidates"], start=len(candidates)):
        candidates.append(
            ImbalanceCandidate(
                name=str(item["name"]),
                strategy="random_oversampling",
                negative_multiplier=int(item["negative_multiplier"]),
                complexity_rank=index,
            )
        )
    return candidates


def build_candidate_pipeline(
    candidate: ImbalanceCandidate,
    model_config: dict[str, Any],
    seed: int,
    fold_train_labels: list[str] | None = None,
) -> Any:
    """Build the frozen feature pipeline, with sampling only inside a fold pipeline."""
    base = build_pipeline(
        ModelSpec("imbalance_experiment", "combined", class_weight=None),
        {**model_config, "random_seed": seed},
    )
    if candidate.strategy == "class_weight":
        base.set_params(classifier__class_weight=candidate.class_weight)
        return base
    if candidate.strategy != "random_oversampling":
        raise ValueError(f"Unsupported imbalance strategy: {candidate.strategy}")
    if fold_train_labels is None or candidate.negative_multiplier is None:
        raise ValueError("Fold labels and multiplier are required for random oversampling")
    negative_count = sum(label == "Negative" for label in fold_train_labels)
    sampler = RandomOverSampler(
        sampling_strategy={"Negative": negative_count * candidate.negative_multiplier},
        random_state=seed,
    )
    return ImbalancedPipeline(
        [
            *base.steps[:-1],
            ("random_oversample", sampler),
            base.steps[-1],
        ]
    )


def threshold_predictions(probabilities: np.ndarray, threshold: str | float) -> list[str]:
    """Apply argmax or an explicit one-vs-rest Negative decision threshold."""
    if threshold == "argmax":
        return [LABELS[int(index)] for index in probabilities.argmax(axis=1)]
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Negative threshold must be between zero and one")
    negative_index = LABELS.index("Negative")
    positive_index = LABELS.index("Positive")
    average_index = LABELS.index("Average")
    predictions = []
    for row in probabilities:
        if row[negative_index] >= value:
            predictions.append("Negative")
        elif row[positive_index] >= row[average_index]:
            predictions.append("Positive")
        else:
            predictions.append("Average")
    return predictions


def _probability_matrix(pipeline: Any, reviews: list[str]) -> np.ndarray:
    raw = pipeline.predict_proba(reviews)
    classes = list(pipeline.named_steps["classifier"].classes_)
    return np.asarray([[row[classes.index(label)] for label in LABELS] for row in raw])


def _language_metrics(
    languages: list[str], labels: list[str], predictions: list[str]
) -> dict[str, dict[str, Any]]:
    result = {}
    for language in sorted(set(languages)):
        indices = [index for index, value in enumerate(languages) if value == language]
        result[language] = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
    return result


def _metric_columns(metrics: dict[str, Any], prefix: str) -> dict[str, float | str]:
    values: dict[str, float | str] = {
        f"{prefix}_{metric}": float(metrics[metric]) for metric in AGGREGATE_METRICS
    }
    for label in LABELS:
        key = label.lower()
        for metric in CLASS_METRICS:
            clean_metric = metric.replace("-", "_")
            values[f"{prefix}_{key}_{clean_metric}"] = float(metrics["per_class"][label][metric])
    values[f"{prefix}_confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
    return values


def evaluate_candidate(
    candidate: ImbalanceCandidate,
    reviews: list[str],
    labels: list[str],
    languages: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    thresholds: list[str | float],
    model_config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    review_array = np.asarray(reviews, dtype=object)
    label_array = np.asarray(labels, dtype=object)
    oof_probabilities = np.zeros((len(labels), len(LABELS)), dtype=float)
    fold_evidence: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
    fit_seconds = []
    for train_indices, validation_indices in folds:
        train_labels = label_array[train_indices].tolist()
        pipeline = build_candidate_pipeline(
            candidate,
            model_config,
            seed,
            fold_train_labels=train_labels,
        )
        started = perf_counter()
        pipeline.fit(review_array[train_indices].tolist(), train_labels)
        fit_seconds.append(perf_counter() - started)
        probabilities = _probability_matrix(pipeline, review_array[validation_indices].tolist())
        oof_probabilities[validation_indices] = probabilities
        fold_evidence.append(
            (validation_indices, probabilities, label_array[validation_indices].tolist())
        )

    rows = []
    class_weight = (
        json.dumps(candidate.class_weight, sort_keys=True)
        if isinstance(candidate.class_weight, dict)
        else candidate.class_weight
    )
    for threshold in thresholds:
        fold_metrics = [
            classification_metrics(true_labels, threshold_predictions(probabilities, threshold))
            for _, probabilities, true_labels in fold_evidence
        ]
        predictions = threshold_predictions(oof_probabilities, threshold)
        overall = classification_metrics(labels, predictions)
        by_language = _language_metrics(languages, labels, predictions)
        threshold_name = "argmax" if threshold == "argmax" else f"{float(threshold):.2f}"
        row: dict[str, Any] = {
            "name": f"{candidate.name}__{threshold_name}",
            "base_candidate": candidate.name,
            "strategy": candidate.strategy,
            "class_weight": class_weight or "none",
            "negative_multiplier": candidate.negative_multiplier or 1,
            "negative_threshold": threshold_name,
            "complexity_rank": candidate.complexity_rank + (threshold != "argmax"),
            "cv_folds": len(folds),
            "cv_fit_seconds": float(sum(fit_seconds)),
            **_metric_columns(overall, "oof"),
        }
        for metric in AGGREGATE_METRICS:
            values = np.asarray([fold[metric] for fold in fold_metrics])
            row[f"cv_{metric}_mean"] = float(values.mean())
            row[f"cv_{metric}_std"] = float(values.std())
        for label in LABELS:
            key = label.lower()
            for metric in CLASS_METRICS:
                clean_metric = metric.replace("-", "_")
                values = np.asarray([fold["per_class"][label][metric] for fold in fold_metrics])
                row[f"cv_{key}_{clean_metric}_mean"] = float(values.mean())
                row[f"cv_{key}_{clean_metric}_std"] = float(values.std())
        for language, metrics in by_language.items():
            row[f"oof_{language}_macro_f1"] = float(metrics["macro_f1"])
            for metric in CLASS_METRICS:
                clean_metric = metric.replace("-", "_")
                row[f"oof_{language}_negative_{clean_metric}"] = float(
                    metrics["per_class"]["Negative"][metric]
                )
        rows.append(row)
    return rows


def select_candidate(results: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Apply precision eligibility, then recall, then documented close-result tie-breaks."""
    selection = config["selection"]
    minimum_precision = float(selection["minimum_negative_precision"])
    tolerance = float(selection["close_negative_recall_tolerance"])
    eligible = results.loc[results["oof_negative_precision"].ge(minimum_precision)].copy()
    if eligible.empty:
        return {
            "eligible": False,
            "selected_candidate": None,
            "reason": f"No candidate reached Negative precision >= {minimum_precision:.2f}",
            "eligible_candidates": 0,
        }
    best_recall = float(eligible["oof_negative_recall"].max())
    close = eligible.loc[eligible["oof_negative_recall"].ge(best_recall - tolerance)].copy()
    selected = close.sort_values(
        ["cv_macro_f1_mean", "cv_macro_f1_std", "complexity_rank", "name"],
        ascending=[False, True, True, True],
    ).iloc[0]
    return {
        "eligible": True,
        "selected_candidate": str(selected["name"]),
        "selected_base_candidate": str(selected["base_candidate"]),
        "selected_threshold": str(selected["negative_threshold"]),
        "negative_precision": float(selected["oof_negative_precision"]),
        "negative_recall": float(selected["oof_negative_recall"]),
        "negative_f1": float(selected["oof_negative_f1_score"]),
        "macro_f1": float(selected["oof_macro_f1"]),
        "cv_macro_f1_mean": float(selected["cv_macro_f1_mean"]),
        "cv_macro_f1_std": float(selected["cv_macro_f1_std"]),
        "eligible_candidates": int(len(eligible)),
        "close_candidates": int(len(close)),
        "maximum_eligible_negative_recall": best_recall,
        "selection_rule": (
            "Negative precision floor; candidates within recall tolerance are ranked by "
            "macro-F1 mean, macro-F1 std, then simplicity"
        ),
    }


def _prepare(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], list[Any]]:
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
    frozen = json.loads(Path(config["frozen_split_metadata"]).read_text(encoding="utf-8"))
    train_hash = _hash_values(split.train["normalized_review"].tolist())
    test_hash = _hash_values(split.test["normalized_review"].tolist())
    if train_hash != frozen["train_normalized_sha256"]:
        raise RuntimeError("Training split does not match frozen submission hash")
    if test_hash != frozen["test_normalized_sha256"]:
        raise RuntimeError("Held-out split does not match frozen submission hash")
    return split, raw, training, [train_hash, test_hash, frozen]


def run_oof(config_path: str | Path) -> dict[str, Any]:
    git_evidence = {
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty_at_start": bool(_git_value("status", "--porcelain")),
    }
    config = load_config(config_path)
    split, raw, training, split_evidence = _prepare(config)
    seed = int(training["random_seed"])
    reviews = split.train["Reviews"].astype(str).tolist()
    labels = split.train["Label"].astype(str).tolist()
    languages = split.train["detected_language"].astype(str).tolist()
    strata = split.train[["detected_language", "Label"]].astype(str).agg("::".join, axis=1)
    cv = StratifiedKFold(n_splits=int(training["cv_folds"]), shuffle=True, random_state=seed)
    folds = list(cv.split(reviews, strata))
    thresholds = list(config["negative_thresholds"])
    rows = []
    for candidate in candidate_specs(config):
        LOGGER.info("OOF probabilities for %s; held-out remains untouched", candidate.name)
        rows.extend(
            evaluate_candidate(
                candidate,
                reviews,
                labels,
                languages,
                folds,
                thresholds,
                training["model"],
                seed,
            )
        )
    results = pd.DataFrame(rows)
    baseline = results.loc[results["name"].eq("balanced__argmax")].iloc[0]
    frozen_comparison = pd.read_csv(config["frozen_cv_comparison"])
    frozen_cv = float(
        frozen_comparison.loc[
            frozen_comparison["name"].eq("combined_balanced_ratings"),
            "cv_macro_f1_mean",
        ].iloc[0]
    )
    if not np.isclose(float(baseline["cv_macro_f1_mean"]), frozen_cv, atol=1e-12):
        raise RuntimeError("Balanced argmax did not reproduce the frozen CV baseline")
    decision = select_candidate(results, config)
    evidence = {
        **git_evidence,
        "heldout_evaluated": False,
        "raw_rows": len(raw),
        "training_rows": len(split.train),
        "test_rows": len(split.test),
        "training_label_counts": split.train["Label"].value_counts().to_dict(),
        "training_language_label_counts": {
            f"{language}::{label}": int(count)
            for (language, label), count in split.train.groupby(["detected_language", "Label"])
            .size()
            .items()
        },
        "train_normalized_sha256": split_evidence[0],
        "test_normalized_sha256": split_evidence[1],
        "frozen_cv_macro_f1": frozen_cv,
        "reproduced_cv_macro_f1": float(baseline["cv_macro_f1_mean"]),
        "base_candidates": len(candidate_specs(config)),
        "decision_rules": len(thresholds),
        "result_rows": len(results),
    }
    output_csv = Path(config["output_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.sort_values(["oof_negative_precision", "oof_negative_recall"], ascending=False).to_csv(
        output_csv, index=False
    )
    payload = {"decision": decision, "evidence": evidence}
    Path(config["output_decision"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def _candidate_from_decision(
    config: dict[str, Any], decision: dict[str, Any]
) -> tuple[ImbalanceCandidate, str | float]:
    base_name = decision["selected_base_candidate"]
    candidate = next(item for item in candidate_specs(config) if item.name == base_name)
    threshold_text = decision["selected_threshold"]
    threshold: str | float = "argmax" if threshold_text == "argmax" else float(threshold_text)
    return candidate, threshold


def _heldout_language_metrics(frame: pd.DataFrame, predictions: list[str]) -> dict[str, Any]:
    return _language_metrics(
        frame["detected_language"].astype(str).tolist(),
        frame["Label"].astype(str).tolist(),
        predictions,
    )


def _metric_comparison(baseline: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for metric in (*AGGREGATE_METRICS,):
        comparison[metric] = {
            "baseline": float(baseline[metric]),
            "selected": float(selected[metric]),
            "delta": float(selected[metric] - baseline[metric]),
        }
    for metric in CLASS_METRICS:
        key = metric.replace("-", "_")
        baseline_value = float(baseline["per_class"]["Negative"][metric])
        selected_value = float(selected["per_class"]["Negative"][metric])
        comparison[f"negative_{key}"] = {
            "baseline": baseline_value,
            "selected": selected_value,
            "delta": selected_value - baseline_value,
        }
    return comparison


def _write_report(
    path: str | Path,
    config: dict[str, Any],
    oof: dict[str, Any],
    heldout: dict[str, Any] | None,
) -> None:
    results = pd.read_csv(config["output_csv"])
    selected_name = oof["decision"].get("selected_candidate")
    selected = results.loc[results["name"].eq(selected_name)] if selected_name else results.head(0)
    baseline = results.loc[results["name"].eq("balanced__argmax")]
    eligible = results.loc[
        results["oof_negative_precision"].ge(
            float(config["selection"]["minimum_negative_precision"])
        )
    ].sort_values(["oof_negative_recall", "cv_macro_f1_mean"], ascending=False)
    display = pd.concat([selected, baseline, eligible.head(8)]).drop_duplicates("name")
    columns = [
        "name",
        "strategy",
        "class_weight",
        "negative_multiplier",
        "negative_threshold",
        "oof_negative_precision",
        "oof_negative_recall",
        "oof_negative_f1_score",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "oof_accuracy",
        "oof_confusion_matrix",
    ]
    oof_table = display[columns].round(4).to_markdown(index=False)
    heldout_section = "Held-out was not evaluated because no OOF candidate was eligible."
    if heldout is not None:
        comparison_rows = [
            {
                "metric": metric,
                **values,
            }
            for metric, values in heldout["comparison"].items()
        ]
        heldout_table = pd.DataFrame(comparison_rows).round(4).to_markdown(index=False)
        per_class_rows = []
        for label in LABELS:
            for metric in CLASS_METRICS:
                baseline_value = float(heldout["baseline"]["per_class"][label][metric])
                selected_value = float(heldout["selected"]["per_class"][label][metric])
                per_class_rows.append(
                    {
                        "label": label,
                        "metric": metric,
                        "baseline": baseline_value,
                        "selected": selected_value,
                        "delta": selected_value - baseline_value,
                    }
                )
        per_class_table = pd.DataFrame(per_class_rows).round(4).to_markdown(index=False)
        heldout_section = f"""{heldout_table}

Per-class held-out metrics:

{per_class_table}

Baseline confusion matrix: `{json.dumps(heldout["baseline"]["confusion_matrix"])}`

Selected confusion matrix: `{json.dumps(heldout["selected"]["confusion_matrix"])}`

Dutch Negative baseline vs selected: precision {heldout["baseline"]["by_language"]["dutch"]["per_class"]["Negative"]["precision"]:.4f} → {heldout["selected"]["by_language"]["dutch"]["per_class"]["Negative"]["precision"]:.4f}; recall {heldout["baseline"]["by_language"]["dutch"]["per_class"]["Negative"]["recall"]:.4f} → {heldout["selected"]["by_language"]["dutch"]["per_class"]["Negative"]["recall"]:.4f}; F1 {heldout["baseline"]["by_language"]["dutch"]["per_class"]["Negative"]["f1-score"]:.4f} → {heldout["selected"]["by_language"]["dutch"]["per_class"]["Negative"]["f1-score"]:.4f}.

Promotion checks:

```json
{json.dumps(heldout["promotion"], indent=2)}
```"""
    recommendation = (
        "Replace the branch candidate only after review; all predeclared promotion gates passed."
        if heldout and heldout["promotion"]["promote"]
        else "Keep `submission-v1`; the candidate did not pass every predeclared promotion gate."
    )
    report = f"""# Negative-Class Imbalance Experiment

## Technical summary

**Recommendation:** {recommendation}

The official baseline remains balanced Logistic Regression with argmax. Candidate weights, fold-local random oversampling, and Negative thresholds were selected only from out-of-fold training predictions. The frozen held-out set was never used to select a weight, sampling ratio, or threshold.

## OOF precision-constrained selection

Eligibility required Negative precision ≥ {float(config["selection"]["minimum_negative_precision"]):.2f}. Candidates within {float(config["selection"]["close_negative_recall_tolerance"]):.2f} recall of the best eligible recall were resolved by macro-F1 mean, macro-F1 standard deviation, and simplicity.

{oof_table}

Full metrics for all {oof["evidence"]["result_rows"]} combinations are in `{config["output_csv"]}`. Each row includes per-class precision/recall/F1, balanced accuracy, macro-F1 mean/std, and an OOF confusion matrix.

A chart is intentionally omitted: the 42-row experiment is an audit table with exact precision/recall constraints, and a reduced visual would hide threshold and fold-dispersion details needed for the decision.

## Frozen held-out comparison

{heldout_section}

The official artifact reports Dutch Negative recall as 31/58 = 0.5345, not 29/58 = 0.5000. Overall Negative recall is 31/60 = 0.5167. These frozen artifact values control the comparison.

## Scope, data, and metric definitions

- Unified training population: {oof["evidence"]["training_rows"]} Dutch and English rows; no language-specific model.
- Training labels: `{json.dumps(oof["evidence"]["training_label_counts"], sort_keys=True)}`.
- Dutch training slice: Positive 1,678; Average 1,540; Negative 232.
- English training slice: Positive 121; Average 259; Negative 8.
- Frozen training hash: `{oof["evidence"]["train_normalized_sha256"]}`.
- Frozen held-out hash: `{oof["evidence"]["test_normalized_sha256"]}`.

## Experimental design and leakage controls

- Five fixed language×label-stratified folds were shared by all candidates.
- Random oversampling occurred after TF-IDF transformation inside each fold's training pipeline only.
- No SMOTE and no majority-class undersampling were used.
- OOF probabilities were generated once per base strategy and reused for threshold comparisons.
- Balanced argmax reproduced frozen CV macro-F1: {oof["evidence"]["reproduced_cv_macro_f1"]:.12f}.
- Held-out lock: `{config["heldout_lock"]}`; it prevents accidental repeat evaluation.

## Limitations and robustness

- Only 240 Negative examples exist in unified training, including just 8 English Negative examples.
- Comparing 42 OOF decision combinations creates selection uncertainty; fold dispersion is reported, and the held-out set is not reused for tuning.
- Weighting, oversampling, and thresholds change decision trade-offs but cannot create missing linguistic coverage.
- Thresholded probabilities are decision scores, not evidence of improved calibration.

## Recommended next steps

1. Keep the frozen submission unless every configured promotion gate passes.
2. For future versions, collect and manually review more real Dutch Negative reviews, especially Negative/Average boundaries, restrained criticism, sarcasm, negation, and mixed praise/criticism.
3. Treat the two held-out English Negative rows as descriptive only; they cannot support a reliable English minority-class conclusion.

## Further questions

- Does the selected policy remain stable after acquiring a materially larger Negative validation sample?
- Which error types account for the remaining Negative→Average mistakes?
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def run_heldout(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    oof = json.loads(Path(config["output_decision"]).read_text(encoding="utf-8"))
    if not oof["decision"]["eligible"]:
        _write_report(config["output_report"], config, oof, None)
        return {"heldout_evaluated": False, "reason": oof["decision"]["reason"]}
    if oof["decision"]["selected_candidate"] == "balanced__argmax":
        baseline = json.loads(Path(config["frozen_baseline_metrics"]).read_text(encoding="utf-8"))
        payload = {
            "candidate": oof["decision"],
            "baseline": baseline,
            "selected": baseline,
            "comparison": _metric_comparison(baseline, baseline),
            "promotion": {
                "checks": {},
                "promote": False,
                "reason": "OOF selected the existing formal baseline",
            },
            "heldout_evaluated_once": False,
            "test_normalized_sha256": oof["evidence"]["test_normalized_sha256"],
        }
        Path(config["output_heldout"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_report(config["output_report"], config, oof, payload)
        return payload
    lock_path = Path(config["heldout_lock"])
    if lock_path.exists():
        raise RuntimeError(f"Held-out evaluation is locked: {lock_path}")
    split, _, training, split_evidence = _prepare(config)
    if oof["evidence"]["train_normalized_sha256"] != split_evidence[0]:
        raise RuntimeError("OOF decision uses a different frozen training split")
    candidate, threshold = _candidate_from_decision(config, oof["decision"])
    train_labels = split.train["Label"].astype(str).tolist()
    pipeline = build_candidate_pipeline(
        candidate,
        training["model"],
        int(training["random_seed"]),
        fold_train_labels=train_labels,
    )
    model = SentimentModel(
        pipeline,
        version=f"negative-candidate+{_git_value('rev-parse', '--short', 'HEAD')}",
        negative_threshold=None if threshold == "argmax" else float(threshold),
    ).fit(split.train["Reviews"].astype(str).tolist(), train_labels)
    model.save(config["output_candidate_model"])
    lock = {
        "status": "started",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "candidate": oof["decision"]["selected_candidate"],
        "test_normalized_sha256": split_evidence[1],
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2)
    test_reviews = split.test["Reviews"].astype(str).tolist()
    probabilities = _probability_matrix(model.pipeline, test_reviews)
    predictions = threshold_predictions(probabilities, threshold)
    test_labels = split.test["Label"].astype(str).tolist()
    selected_metrics = classification_metrics(test_labels, predictions)
    selected_metrics["by_language"] = _heldout_language_metrics(split.test, predictions)
    baseline_metrics = json.loads(
        Path(config["frozen_baseline_metrics"]).read_text(encoding="utf-8")
    )
    comparison = _metric_comparison(baseline_metrics, selected_metrics)
    promotion_config = config["promotion"]
    checks = {
        "negative_precision_floor": selected_metrics["per_class"]["Negative"]["precision"]
        >= float(promotion_config["minimum_heldout_negative_precision"]),
        "negative_recall_improved": comparison["negative_recall"]["delta"]
        > float(promotion_config["minimum_heldout_negative_recall_improvement"]),
        "macro_f1_guardrail": comparison["macro_f1"]["delta"]
        >= -float(promotion_config["maximum_heldout_macro_f1_drop"]),
        "accuracy_guardrail": comparison["accuracy"]["delta"]
        >= -float(promotion_config["maximum_heldout_accuracy_drop"]),
    }
    promotion = {"checks": checks, "promote": all(checks.values())}
    payload = {
        "candidate": oof["decision"],
        "baseline": baseline_metrics,
        "selected": selected_metrics,
        "comparison": comparison,
        "promotion": promotion,
        "heldout_evaluated_once": True,
        "test_normalized_sha256": split_evidence[1],
    }
    Path(config["output_heldout"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lock["status"] = "complete"
    lock["completed_at_utc"] = datetime.now(UTC).isoformat()
    lock["promotion"] = promotion
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    _write_report(config["output_report"], config, oof, payload)
    print(json.dumps({"comparison": comparison, "promotion": promotion}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/imbalance_experiment.yaml")
    parser.add_argument("--stage", choices=("oof", "heldout"), default="oof")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "oof":
        run_oof(args.config)
    else:
        run_heldout(args.config)


if __name__ == "__main__":
    main()
