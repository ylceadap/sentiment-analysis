"""Evaluate two-boundary ordinal logistic classifiers using training-only OOF evidence."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import load_config
from .data import annotate_review_languages, load_dataset, make_holdout_split, sha256_file
from .language import DutchLanguageDetector
from .metrics import classification_metrics
from .model import ModelSpec, build_pipeline

LOGGER = logging.getLogger(__name__)
ORDERED_LABELS = ("Negative", "Average", "Positive")


def project_monotonic_boundaries(
    above_negative: np.ndarray, above_average: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Project independent cumulative probabilities onto p(y>N) >= p(y>A)."""
    lower = np.asarray(above_negative, dtype=float).copy()
    upper = np.asarray(above_average, dtype=float).copy()
    if lower.shape != upper.shape:
        raise ValueError("Ordinal boundary probability arrays must have the same shape")
    violations = upper > lower
    original_gap = np.maximum(upper - lower, 0.0)
    midpoint = (lower[violations] + upper[violations]) / 2.0
    lower[violations] = midpoint
    upper[violations] = midpoint
    evidence = {
        "violation_rate": float(violations.mean()) if len(violations) else 0.0,
        "mean_violation_gap": float(original_gap.mean()) if len(original_gap) else 0.0,
    }
    return lower, upper, evidence


def compose_ordinal_probabilities(
    above_negative: np.ndarray, above_average: np.ndarray
) -> np.ndarray:
    """Compose ordered Negative/Average/Positive probabilities from two boundaries."""
    lower, upper, _ = project_monotonic_boundaries(above_negative, above_average)
    probabilities = np.column_stack((1.0 - lower, lower - upper, upper))
    probabilities = np.clip(probabilities, 0.0, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("Ordinal probability composition produced an empty row")
    return probabilities / row_sums


def probability_argmax_labels(probabilities: np.ndarray) -> list[str]:
    """Return labels from Negative/Average/Positive ordered probability columns."""
    return [ORDERED_LABELS[int(index)] for index in np.argmax(probabilities, axis=1)]


def boundary_threshold_labels(
    above_negative: np.ndarray,
    above_average: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> list[str]:
    """Apply sequential ordered decisions using independently tunable boundaries."""
    if not 0 < lower_threshold < 1 or not 0 < upper_threshold < 1:
        raise ValueError("Ordinal decision thresholds must lie strictly between zero and one")
    lower, upper, _ = project_monotonic_boundaries(above_negative, above_average)
    predictions = np.where(
        lower < lower_threshold,
        "Negative",
        np.where(upper >= upper_threshold, "Positive", "Average"),
    )
    return predictions.tolist()


def _negative_metrics(metrics: dict[str, Any]) -> tuple[float, float, float]:
    values = metrics["per_class"]["Negative"]
    return float(values["precision"]), float(values["recall"]), float(values["f1-score"])


def _passes_gate(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    precision, recall, _ = _negative_metrics(metrics)
    _, baseline_recall, _ = _negative_metrics(baseline)
    return bool(
        precision >= 0.60
        and recall >= baseline_recall
        and metrics["macro_f1"] >= baseline["macro_f1"] - 0.01
    )


def _passes_blind_evaluation_gate(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        _passes_gate(metrics, baseline)
        and metrics["ordinal_mae"] < baseline["ordinal_mae"]
        and metrics["quadratic_weighted_kappa"] > baseline["quadratic_weighted_kappa"]
        and metrics["severe_error_rate"] < baseline["severe_error_rate"]
    )


def _selection_rank(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    _, negative_recall, _ = _negative_metrics(metrics)
    return (
        -negative_recall,
        -metrics["macro_f1"],
        metrics["ordinal_mae"],
        metrics["severe_error_rate"],
        -metrics["quadratic_weighted_kappa"],
    )


def _select_thresholds(
    labels: list[str],
    above_negative: np.ndarray,
    above_average: np.ndarray,
    baseline: dict[str, Any],
    threshold_values: list[float],
) -> tuple[float, float, dict[str, Any]]:
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for lower_threshold in threshold_values:
        for upper_threshold in threshold_values:
            predictions = boundary_threshold_labels(
                above_negative,
                above_average,
                lower_threshold,
                upper_threshold,
            )
            metrics = classification_metrics(labels, predictions)
            candidates.append((lower_threshold, upper_threshold, metrics))
    blind_ready = [
        candidate
        for candidate in candidates
        if _passes_blind_evaluation_gate(candidate[2], baseline)
    ]
    if blind_ready:
        return min(blind_ready, key=lambda candidate: _selection_rank(candidate[2]))
    eligible = [candidate for candidate in candidates if _passes_gate(candidate[2], baseline)]
    if eligible:
        return min(eligible, key=lambda candidate: _selection_rank(candidate[2]))
    return max(candidates, key=lambda candidate: candidate[2]["macro_f1"])


def _fold_summary(
    labels: list[str], predictions: list[str], fold_assignments: np.ndarray
) -> dict[str, float]:
    macro_f1_values: list[float] = []
    balanced_accuracy_values: list[float] = []
    for fold in sorted(set(fold_assignments.tolist())):
        indices = np.flatnonzero(fold_assignments == fold)
        metrics = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
        macro_f1_values.append(metrics["macro_f1"])
        balanced_accuracy_values.append(metrics["balanced_accuracy"])
    return {
        "cv_macro_f1_mean": float(np.mean(macro_f1_values)),
        "cv_macro_f1_std": float(np.std(macro_f1_values)),
        "cv_balanced_accuracy_mean": float(np.mean(balanced_accuracy_values)),
        "cv_balanced_accuracy_std": float(np.std(balanced_accuracy_values)),
    }


def _cross_fitted_threshold_predictions(
    labels: list[str],
    above_negative: np.ndarray,
    above_average: np.ndarray,
    baseline_predictions: list[str],
    fold_assignments: np.ndarray,
    threshold_values: list[float],
) -> tuple[list[str], list[dict[str, float]]]:
    """Select thresholds away from each fold before evaluating that fold."""
    predictions = np.empty(len(labels), dtype=object)
    threshold_evidence: list[dict[str, float]] = []
    for fold in sorted(set(fold_assignments.tolist())):
        tuning_indices = np.flatnonzero(fold_assignments != fold)
        evaluation_indices = np.flatnonzero(fold_assignments == fold)
        tuning_labels = [labels[index] for index in tuning_indices]
        tuning_baseline = classification_metrics(
            tuning_labels, [baseline_predictions[index] for index in tuning_indices]
        )
        lower_threshold, upper_threshold, _ = _select_thresholds(
            tuning_labels,
            above_negative[tuning_indices],
            above_average[tuning_indices],
            tuning_baseline,
            threshold_values,
        )
        predictions[evaluation_indices] = boundary_threshold_labels(
            above_negative[evaluation_indices],
            above_average[evaluation_indices],
            lower_threshold,
            upper_threshold,
        )
        threshold_evidence.append(
            {
                "fold": float(fold),
                "lower_threshold": lower_threshold,
                "upper_threshold": upper_threshold,
            }
        )
    return predictions.tolist(), threshold_evidence


def _candidate_record(
    name: str,
    kind: str,
    labels: list[str],
    predictions: list[str],
    fold_assignments: np.ndarray,
    parameters: dict[str, Any],
    probability_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "parameters": parameters,
        "metrics": classification_metrics(labels, predictions),
        "fold_summary": _fold_summary(labels, predictions, fold_assignments),
        "probability_evidence": probability_evidence,
    }


def _flat_row(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate["metrics"]
    precision, recall, f1 = _negative_metrics(metrics)
    return {
        "name": candidate["name"],
        "kind": candidate["kind"],
        **candidate["parameters"],
        **candidate["fold_summary"],
        "oof_macro_f1": metrics["macro_f1"],
        "oof_balanced_accuracy": metrics["balanced_accuracy"],
        "negative_precision": precision,
        "negative_recall": recall,
        "negative_f1": f1,
        "ordinal_mae": metrics["ordinal_mae"],
        "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
        "adjacent_error_rate": metrics["adjacent_error_rate"],
        "severe_error_rate": metrics["severe_error_rate"],
    }


def _calibrated_boundary_classifier(
    *, c_value: float, seed: int, max_iter: int, calibration_folds: int
) -> CalibratedClassifierCV:
    estimator = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=calibration_folds, n_jobs=1)


def run_ordinal_logistic_experiment(config_path: str | Path) -> dict[str, Any]:
    """Run leakage-safe OOF ordinal boundary experiments without evaluating held-out rows."""
    config = load_config(config_path)
    seed = int(config["random_seed"])
    experiment_config = config.get("ordinal_logistic", {})
    output_dir = Path(experiment_config.get("output_dir", "artifacts/ordinal_logistic"))
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(config["data_path"])
    detector = DutchLanguageDetector(**config["language"])
    annotated, _ = annotate_review_languages(raw, detector)
    split = make_holdout_split(
        annotated,
        test_size=float(config["test_size"]),
        random_seed=seed,
        stratify_columns=("detected_language", "Label"),
    )
    reviews = split.train["Reviews"].astype(str).tolist()
    labels = split.train["Label"].astype(str).tolist()
    labels_array = np.asarray(labels)
    strata = split.train[["detected_language", "Label"]].astype(str).agg("::".join, axis=1)
    fold_iterator = StratifiedKFold(
        n_splits=int(config["cv_folds"]), shuffle=True, random_state=seed
    ).split(reviews, strata)

    model_config = {**config["model"], "random_seed": seed}
    base_pipeline = build_pipeline(
        ModelSpec("combined_balanced_ratings", "combined", "balanced"), model_config
    )
    c_values = [float(value) for value in experiment_config.get("c_values", [0.5, 1.0, 2.0])]
    calibration_folds = int(experiment_config.get("calibration_folds", 3))
    max_iter = int(model_config.get("max_iter", 1500))

    baseline_predictions = np.empty(len(labels), dtype=object)
    fold_assignments = np.full(len(labels), -1, dtype=int)
    boundary_probabilities = {
        c_value: {
            "above_negative": np.zeros(len(labels), dtype=float),
            "above_average": np.zeros(len(labels), dtype=float),
        }
        for c_value in c_values
    }

    for fold_number, (train_indices, validation_indices) in enumerate(fold_iterator):
        LOGGER.info("Fitting ordinal-logistic OOF fold %d", fold_number + 1)
        fold_assignments[validation_indices] = fold_number
        feature_pipeline = Pipeline(base_pipeline.steps[:-1])
        train_reviews = [reviews[index] for index in train_indices]
        validation_reviews = [reviews[index] for index in validation_indices]
        x_train = feature_pipeline.fit_transform(train_reviews)
        x_validation = feature_pipeline.transform(validation_reviews)

        baseline = clone(base_pipeline.named_steps["classifier"])
        baseline.fit(x_train, labels_array[train_indices])
        baseline_predictions[validation_indices] = baseline.predict(x_validation)

        lower_targets = (labels_array[train_indices] != "Negative").astype(int)
        upper_targets = (labels_array[train_indices] == "Positive").astype(int)
        for c_value in c_values:
            lower_classifier = _calibrated_boundary_classifier(
                c_value=c_value,
                seed=seed,
                max_iter=max_iter,
                calibration_folds=calibration_folds,
            )
            upper_classifier = _calibrated_boundary_classifier(
                c_value=c_value,
                seed=seed,
                max_iter=max_iter,
                calibration_folds=calibration_folds,
            )
            lower_classifier.fit(x_train, lower_targets)
            upper_classifier.fit(x_train, upper_targets)
            boundary_probabilities[c_value]["above_negative"][validation_indices] = (
                lower_classifier.predict_proba(x_validation)[:, 1]
            )
            boundary_probabilities[c_value]["above_average"][validation_indices] = (
                upper_classifier.predict_proba(x_validation)[:, 1]
            )

    if np.any(fold_assignments < 0):
        raise RuntimeError("Some training rows did not receive OOF predictions")
    baseline_candidate = _candidate_record(
        "baseline_multiclass",
        "multiclass_logistic_regression",
        labels,
        baseline_predictions.tolist(),
        fold_assignments,
        {"class_weight": "balanced", "C": 1.0},
    )
    baseline_metrics = baseline_candidate["metrics"]
    candidates = [baseline_candidate]
    threshold_values = [
        float(value)
        for value in experiment_config.get(
            "threshold_values",
            [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        )
    ]

    for c_value in c_values:
        raw_lower = boundary_probabilities[c_value]["above_negative"]
        raw_upper = boundary_probabilities[c_value]["above_average"]
        lower, upper, probability_evidence = project_monotonic_boundaries(raw_lower, raw_upper)
        composed = compose_ordinal_probabilities(lower, upper)
        candidates.append(
            _candidate_record(
                f"ordinal_composed_argmax_C_{c_value:g}",
                "two_boundary_composed_argmax",
                labels,
                probability_argmax_labels(composed),
                fold_assignments,
                {
                    "C": c_value,
                    "class_weight": "balanced",
                    "calibration": "sigmoid",
                },
                probability_evidence,
            )
        )
        deployment_lower_threshold, deployment_upper_threshold, _ = _select_thresholds(
            labels,
            lower,
            upper,
            baseline_metrics,
            threshold_values,
        )
        threshold_predictions, thresholds_by_fold = _cross_fitted_threshold_predictions(
            labels,
            lower,
            upper,
            baseline_predictions.tolist(),
            fold_assignments,
            threshold_values,
        )
        threshold_candidate = _candidate_record(
            f"ordinal_crossfit_threshold_C_{c_value:g}",
            "two_boundary_crossfit_threshold",
            labels,
            threshold_predictions,
            fold_assignments,
            {
                "C": c_value,
                "class_weight": "balanced",
                "calibration": "sigmoid",
                "threshold_selection": "cross_fitted_oof",
                "deployment_lower_threshold": deployment_lower_threshold,
                "deployment_upper_threshold": deployment_upper_threshold,
            },
            {**probability_evidence, "thresholds_by_fold": thresholds_by_fold},
        )
        candidates.append(threshold_candidate)

    blind_ready = [
        candidate
        for candidate in candidates[1:]
        if _passes_blind_evaluation_gate(candidate["metrics"], baseline_metrics)
    ]
    eligible = [
        candidate
        for candidate in candidates[1:]
        if _passes_gate(candidate["metrics"], baseline_metrics)
    ]
    selected = (
        min(blind_ready, key=lambda candidate: _selection_rank(candidate["metrics"]))
        if blind_ready
        else min(eligible, key=lambda candidate: _selection_rank(candidate["metrics"]))
        if eligible
        else baseline_candidate
    )
    selected_metrics = selected["metrics"]
    blind_evaluation_recommended = _passes_blind_evaluation_gate(selected_metrics, baseline_metrics)
    result = {
        "status": "training_oof_only",
        "held_out_test_evaluated": False,
        "method": {
            "lower_boundary": "Negative vs Average+Positive",
            "upper_boundary": "Negative+Average vs Positive",
            "class_weight": "balanced",
            "calibration": "fold-internal sigmoid",
            "monotonic_projection": "pool violating boundary probabilities at their midpoint",
            "threshold_evaluation": (
                "cross-fitted: each fold uses thresholds selected on the other OOF folds"
            ),
        },
        "selection_gate": {
            "negative_precision_minimum": 0.60,
            "negative_recall_minimum": baseline_metrics["per_class"]["Negative"]["recall"],
            "macro_f1_maximum_drop": 0.01,
            "eligible_candidate_priority": [
                "negative_recall_descending",
                "macro_f1_descending",
                "ordinal_mae_ascending",
                "severe_error_rate_ascending",
                "quadratic_weighted_kappa_descending",
            ],
            "promotion_requires_lower_ordinal_mae": True,
            "promotion_requires_higher_quadratic_weighted_kappa": True,
            "promotion_requires_lower_severe_error_rate": True,
        },
        "data": {
            "raw_sha256": sha256_file(config["data_path"]),
            "oof_training_rows": len(labels),
            "reserved_held_out_rows": len(split.test),
            "cv_folds": int(config["cv_folds"]),
            "calibration_folds": calibration_folds,
            "random_seed": seed,
        },
        "selected_candidate": selected["name"],
        "oof_gate_passed": selected["name"] != baseline_candidate["name"],
        "new_blind_evaluation_recommended": blind_evaluation_recommended,
        "production_promotion_recommended": False,
        "production_promotion_blocker": (
            "A new untouched blind evaluation is required before replacing the submitted model."
        ),
        "candidates": {candidate["name"]: candidate for candidate in candidates},
    }
    (output_dir / "ordinal_logistic_experiment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame([_flat_row(candidate) for candidate in candidates]).sort_values(
        ["severe_error_rate", "ordinal_mae", "oof_macro_f1"], ascending=[True, True, False]
    ).to_csv(output_dir / "ordinal_logistic_experiment.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_ordinal_logistic_experiment(args.config)
    LOGGER.info(
        "Selected %s; new blind evaluation recommended: %s",
        result["selected_candidate"],
        result["new_blind_evaluation_recommended"],
    )


if __name__ == "__main__":
    main()
