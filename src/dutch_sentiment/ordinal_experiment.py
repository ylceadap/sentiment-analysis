"""Compare nominal and ordinal sentiment decisions using training-only OOF evidence."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .config import load_config
from .constants import LABELS
from .data import annotate_review_languages, load_dataset, make_holdout_split, sha256_file
from .language import DutchLanguageDetector
from .metrics import ORDINAL_VALUES, classification_metrics
from .model import ModelSpec, build_pipeline

LOGGER = logging.getLogger(__name__)
VALUE_TO_LABEL = {value: label for label, value in ORDINAL_VALUES.items()}


def scores_to_labels(scores: np.ndarray, lower: float, upper: float) -> list[str]:
    """Convert continuous 1/2/3 sentiment scores into ordered labels."""
    if lower >= upper:
        raise ValueError("The lower ordinal threshold must be below the upper threshold")
    values = np.where(scores < lower, 1, np.where(scores < upper, 2, 3))
    return [VALUE_TO_LABEL[int(value)] for value in values]


def cost_sensitive_labels(
    probabilities: np.ndarray,
    probability_labels: list[str],
    severe_cost: float,
) -> list[str]:
    """Minimize expected ordinal cost, with larger cost for two-level mistakes."""
    if severe_cost < 1:
        raise ValueError("Severe error cost must be at least one")
    ordered_labels = ["Negative", "Average", "Positive"]
    ordered_probabilities = probabilities[
        :, [probability_labels.index(label) for label in ordered_labels]
    ]
    costs = np.asarray(
        [
            [0.0, 1.0, severe_cost],
            [1.0, 0.0, 1.0],
            [severe_cost, 1.0, 0.0],
        ]
    )
    predicted_indices = np.argmin(ordered_probabilities @ costs, axis=1)
    return [ordered_labels[int(index)] for index in predicted_indices]


def _negative_metrics(metrics: dict[str, Any]) -> tuple[float, float, float]:
    negative = metrics["per_class"]["Negative"]
    return float(negative["precision"]), float(negative["recall"]), float(negative["f1-score"])


def _passes_gate(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    precision, recall, _ = _negative_metrics(metrics)
    _, baseline_recall, _ = _negative_metrics(baseline)
    return bool(
        precision >= 0.60
        and recall >= baseline_recall
        and metrics["macro_f1"] >= baseline["macro_f1"] - 0.01
    )


def _ordinal_rank(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        metrics["severe_error_rate"],
        metrics["ordinal_mae"],
        -metrics["quadratic_weighted_kappa"],
        -metrics["macro_f1"],
    )


def _select_thresholds(
    labels: list[str],
    scores: np.ndarray,
    baseline: dict[str, Any],
    threshold_values: list[float],
) -> tuple[float, float, dict[str, Any]]:
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for lower in threshold_values:
        for upper in threshold_values:
            if lower >= upper:
                continue
            predictions = scores_to_labels(scores, lower, upper)
            metrics = classification_metrics(labels, predictions)
            candidates.append((lower, upper, metrics))
    eligible = [candidate for candidate in candidates if _passes_gate(candidate[2], baseline)]
    pool = eligible or candidates
    if eligible:
        return min(pool, key=lambda candidate: _ordinal_rank(candidate[2]))
    return max(pool, key=lambda candidate: candidate[2]["macro_f1"])


def _flat_row(name: str, kind: str, metrics: dict[str, Any], **params: Any) -> dict[str, Any]:
    precision, recall, f1 = _negative_metrics(metrics)
    return {
        "name": name,
        "kind": kind,
        **params,
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "negative_precision": precision,
        "negative_recall": recall,
        "negative_f1": f1,
        "ordinal_mae": metrics["ordinal_mae"],
        "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
        "adjacent_error_rate": metrics["adjacent_error_rate"],
        "severe_error_rate": metrics["severe_error_rate"],
    }


def run_ordinal_experiment(config_path: str | Path) -> dict[str, Any]:
    """Run nominal, cost-aware, and continuous-score models without test evaluation."""
    config = load_config(config_path)
    seed = int(config["random_seed"])
    ordinal_config = config.get("ordinal", {})
    output_dir = Path(ordinal_config.get("output_dir", "artifacts/ordinal"))
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
    strata = split.train[["detected_language", "Label"]].astype(str).agg("::".join, axis=1)
    folds = StratifiedKFold(
        n_splits=int(config["cv_folds"]), shuffle=True, random_state=seed
    ).split(reviews, strata)

    base_pipeline = build_pipeline(
        ModelSpec("combined_balanced_ratings", "combined", "balanced"),
        {**config["model"], "random_seed": seed},
    )
    alphas = [float(value) for value in ordinal_config.get("ridge_alphas", [0.1, 1.0, 10.0])]
    weighting_options = [False, True]
    baseline_predictions = np.empty(len(labels), dtype=object)
    baseline_probabilities = np.zeros((len(labels), len(LABELS)), dtype=float)
    baseline_probability_labels: list[str] | None = None
    ridge_scores = {
        (alpha, weighted): np.zeros(len(labels), dtype=float)
        for alpha in alphas
        for weighted in weighting_options
    }
    ordinal_targets = np.asarray([ORDINAL_VALUES[label] for label in labels], dtype=float)

    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        LOGGER.info("Fitting ordinal OOF fold %d", fold_number)
        feature_pipeline = Pipeline(base_pipeline.steps[:-1])
        train_reviews = [reviews[index] for index in train_indices]
        validation_reviews = [reviews[index] for index in validation_indices]
        x_train = feature_pipeline.fit_transform(train_reviews)
        x_validation = feature_pipeline.transform(validation_reviews)

        classifier = clone(base_pipeline.named_steps["classifier"])
        classifier.fit(x_train, np.asarray(labels)[train_indices])
        baseline_predictions[validation_indices] = classifier.predict(x_validation)
        current_probability_labels = [str(label) for label in classifier.classes_]
        if baseline_probability_labels is None:
            baseline_probability_labels = current_probability_labels
        elif current_probability_labels != baseline_probability_labels:
            raise RuntimeError("Classifier class order changed across folds")
        baseline_probabilities[validation_indices] = classifier.predict_proba(x_validation)

        sample_weights = compute_sample_weight("balanced", np.asarray(labels)[train_indices])
        for alpha in alphas:
            for weighted in weighting_options:
                regressor = Ridge(alpha=alpha, solver="lsqr")
                regressor.fit(
                    x_train,
                    ordinal_targets[train_indices],
                    sample_weight=sample_weights if weighted else None,
                )
                ridge_scores[(alpha, weighted)][validation_indices] = regressor.predict(
                    x_validation
                )

    if baseline_probability_labels is None:
        raise RuntimeError("No OOF folds were produced")
    baseline_metrics = classification_metrics(labels, baseline_predictions.tolist())
    detailed: dict[str, dict[str, Any]] = {
        "baseline_argmax": {
            "kind": "multiclass_logistic_regression",
            "parameters": {},
            "metrics": baseline_metrics,
        }
    }
    rows = [_flat_row("baseline_argmax", "multiclass_logistic_regression", baseline_metrics)]

    for severe_cost in ordinal_config.get("severe_costs", [2, 3, 4]):
        predictions = cost_sensitive_labels(
            baseline_probabilities, baseline_probability_labels, float(severe_cost)
        )
        metrics = classification_metrics(labels, predictions)
        name = f"cost_sensitive_severe_{severe_cost}"
        detailed[name] = {
            "kind": "cost_sensitive_decision",
            "parameters": {"adjacent_cost": 1, "severe_cost": severe_cost},
            "metrics": metrics,
        }
        rows.append(
            _flat_row(
                name,
                "cost_sensitive_decision",
                metrics,
                severe_cost=severe_cost,
            )
        )

    threshold_values = [
        float(value)
        for value in ordinal_config.get(
            "threshold_values", np.round(np.arange(1.2, 2.81, 0.05), 2).tolist()
        )
    ]
    for (alpha, weighted), scores in ridge_scores.items():
        lower, upper, metrics = _select_thresholds(
            labels, scores, baseline_metrics, threshold_values
        )
        weighting = "balanced" if weighted else "unweighted"
        name = f"ridge_{weighting}_alpha_{alpha:g}"
        detailed[name] = {
            "kind": "ordinal_ridge_regression",
            "parameters": {
                "label_values": ORDINAL_VALUES,
                "alpha": alpha,
                "class_weighting": weighting,
                "lower_threshold": lower,
                "upper_threshold": upper,
            },
            "metrics": metrics,
            "continuous_score_summary": {
                "minimum": float(scores.min()),
                "mean": float(scores.mean()),
                "maximum": float(scores.max()),
            },
        }
        rows.append(
            _flat_row(
                name,
                "ordinal_ridge_regression",
                metrics,
                alpha=alpha,
                class_weighting=weighting,
                lower_threshold=lower,
                upper_threshold=upper,
            )
        )

    eligible_names = [
        name
        for name, candidate in detailed.items()
        if name != "baseline_argmax" and _passes_gate(candidate["metrics"], baseline_metrics)
    ]
    best_name = (
        min(eligible_names, key=lambda name: _ordinal_rank(detailed[name]["metrics"]))
        if eligible_names
        else "baseline_argmax"
    )
    best_metrics = detailed[best_name]["metrics"]
    promote = bool(
        best_name != "baseline_argmax"
        and best_metrics["ordinal_mae"] < baseline_metrics["ordinal_mae"]
        and best_metrics["quadratic_weighted_kappa"] > baseline_metrics["quadratic_weighted_kappa"]
        and best_metrics["severe_error_rate"] < baseline_metrics["severe_error_rate"]
    )
    result = {
        "status": "training_oof_only",
        "held_out_test_evaluated": False,
        "label_values": ORDINAL_VALUES,
        "selection_gate": {
            "negative_precision_minimum": 0.60,
            "negative_recall_minimum": baseline_metrics["per_class"]["Negative"]["recall"],
            "macro_f1_maximum_drop": 0.01,
            "promotion_requires_lower_ordinal_mae": True,
            "promotion_requires_higher_quadratic_weighted_kappa": True,
            "promotion_requires_lower_severe_error_rate": True,
        },
        "data": {
            "raw_sha256": sha256_file(config["data_path"]),
            "oof_training_rows": len(labels),
            "reserved_held_out_rows": len(split.test),
            "cv_folds": int(config["cv_folds"]),
            "random_seed": seed,
        },
        "selected_candidate": best_name,
        "promotion_recommended": promote,
        "candidates": detailed,
    }
    (output_dir / "ordinal_experiment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).sort_values(
        ["severe_error_rate", "ordinal_mae", "macro_f1"], ascending=[True, True, False]
    ).to_csv(output_dir / "ordinal_experiment.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_ordinal_experiment(args.config)
    LOGGER.info(
        "Selected %s; promotion recommended: %s",
        result["selected_candidate"],
        result["promotion_recommended"],
    )


if __name__ == "__main__":
    main()
