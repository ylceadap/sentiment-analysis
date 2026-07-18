"""Train-only Jina embedding plus ordinal-logistic OOF experiment."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from .data import annotate_review_languages, load_dataset, make_holdout_split, sha256_file
from .embedding_experiment import (
    _aligned_probabilities,
    _encode_or_load,
    _hash_values,
    _official_baseline,
    threshold_predictions,
)
from .language import DutchLanguageDetector
from .metrics import classification_metrics

LOGGER = logging.getLogger(__name__)
ORDERED_LABELS = ("Negative", "Average", "Positive")
ORDINAL_VALUES: dict[str, int] = {"Negative": 1, "Average": 2, "Positive": 3}


def project_monotonic_boundaries(
    above_negative: np.ndarray, above_average: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    lower = np.asarray(above_negative, dtype=float).copy()
    upper = np.asarray(above_average, dtype=float).copy()
    if lower.shape != upper.shape:
        raise ValueError("Ordinal boundary probability arrays must have the same shape")
    violations = upper > lower
    original_gap = np.maximum(upper - lower, 0.0)
    midpoint = (lower[violations] + upper[violations]) / 2.0
    lower[violations] = midpoint
    upper[violations] = midpoint
    return (
        lower,
        upper,
        {
            "violation_rate": float(violations.mean()) if len(violations) else 0.0,
            "mean_violation_gap": float(original_gap.mean()) if len(original_gap) else 0.0,
        },
    )


def compose_ordinal_probabilities(
    above_negative: np.ndarray, above_average: np.ndarray
) -> np.ndarray:
    lower, upper, _ = project_monotonic_boundaries(above_negative, above_average)
    probabilities = np.column_stack((1.0 - lower, lower - upper, upper))
    probabilities = np.clip(probabilities, 0.0, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("Ordinal probability composition produced an empty row")
    return probabilities / row_sums


def probability_argmax_labels(probabilities: np.ndarray) -> list[str]:
    return [ORDERED_LABELS[int(index)] for index in np.argmax(probabilities, axis=1)]


def boundary_threshold_labels(
    above_negative: np.ndarray,
    above_average: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> list[str]:
    if not 0 < lower_threshold < 1 or not 0 < upper_threshold < 1:
        raise ValueError("Ordinal decision thresholds must lie strictly between zero and one")
    lower, upper, _ = project_monotonic_boundaries(above_negative, above_average)
    predictions = np.where(
        lower < lower_threshold,
        "Negative",
        np.where(upper >= upper_threshold, "Positive", "Average"),
    )
    return predictions.tolist()


def _with_ordinal_diagnostics(metrics: dict[str, Any], labels: list[str], predictions: list[str]):
    true_ordinal = np.asarray([ORDINAL_VALUES[label] for label in labels])
    predicted_ordinal = np.asarray([ORDINAL_VALUES[label] for label in predictions])
    distance = np.abs(true_ordinal - predicted_ordinal)
    qwk = float(cohen_kappa_score(true_ordinal, predicted_ordinal, weights="quadratic"))
    if not np.isfinite(qwk):
        qwk = 1.0 if np.array_equal(true_ordinal, predicted_ordinal) else 0.0
    return {
        **metrics,
        "ordinal_mae": float(distance.mean()),
        "quadratic_weighted_kappa": qwk,
        "adjacent_error_rate": float(np.mean(distance == 1)),
        "severe_error_rate": float(np.mean(distance == 2)),
    }


def _negative_metrics(metrics: dict[str, Any]) -> tuple[float, float, float]:
    negative = metrics["per_class"]["Negative"]
    return (
        float(negative["precision"]),
        float(negative["recall"]),
        float(negative["f1-score"]),
    )


def _fold_summary(
    labels: list[str], predictions: list[str], validation_folds: list[np.ndarray]
) -> dict[str, float]:
    fold_metrics = [
        classification_metrics(
            [labels[index] for index in fold],
            [predictions[index] for index in fold],
        )
        for fold in validation_folds
    ]
    return {
        "cv_macro_f1_mean": float(np.mean([item["macro_f1"] for item in fold_metrics])),
        "cv_macro_f1_std": float(np.std([item["macro_f1"] for item in fold_metrics])),
        "cv_balanced_accuracy_mean": float(
            np.mean([item["balanced_accuracy"] for item in fold_metrics])
        ),
        "cv_balanced_accuracy_std": float(
            np.std([item["balanced_accuracy"] for item in fold_metrics])
        ),
        "cv_accuracy_mean": float(np.mean([item["accuracy"] for item in fold_metrics])),
        "cv_accuracy_std": float(np.std([item["accuracy"] for item in fold_metrics])),
    }


def _language_slices(
    labels: list[str], predictions: list[str], languages: list[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for language in sorted(set(languages)):
        indices = [index for index, value in enumerate(languages) if value == language]
        sliced = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
        result[f"{language}_rows"] = float(len(indices))
        result[f"{language}_macro_f1"] = sliced["macro_f1"]
        result[f"{language}_accuracy"] = sliced["accuracy"]
        result[f"{language}_negative_recall"] = sliced["per_class"]["Negative"]["recall"]
    return result


def _candidate_row(
    *,
    name: str,
    model_type: str,
    model_id: str,
    revision: str,
    c_value: float,
    decision_rule: str,
    labels: list[str],
    languages: list[str],
    predictions: list[str],
    validation_folds: list[np.ndarray],
    parameters: dict[str, Any],
    runtime: dict[str, Any],
    probability_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _with_ordinal_diagnostics(
        classification_metrics(labels, predictions), labels, predictions
    )
    precision, recall, f1 = _negative_metrics(metrics)
    return {
        "name": name,
        "model_type": model_type,
        "model_id": model_id,
        "revision": revision,
        "C": c_value,
        "decision_rule": decision_rule,
        **parameters,
        **_fold_summary(labels, predictions, validation_folds),
        "oof_macro_f1": metrics["macro_f1"],
        "oof_balanced_accuracy": metrics["balanced_accuracy"],
        "oof_accuracy": metrics["accuracy"],
        "negative_precision": precision,
        "negative_recall": recall,
        "negative_f1": f1,
        "ordinal_mae": metrics["ordinal_mae"],
        "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
        "adjacent_error_rate": metrics["adjacent_error_rate"],
        "severe_error_rate": metrics["severe_error_rate"],
        "per_class_json": json.dumps(metrics["per_class"], sort_keys=True),
        "confusion_matrix_json": json.dumps(metrics["confusion_matrix"]),
        "probability_evidence_json": json.dumps(probability_evidence or {}, sort_keys=True),
        **_language_slices(labels, predictions, languages),
        **runtime,
    }


def _calibrated_boundary_classifier(
    *, c_value: float, seed: int, calibration_folds: int
) -> CalibratedClassifierCV:
    estimator = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=calibration_folds, n_jobs=1)


def _embedding_multiclass_rows(
    model_spec: dict[str, Any],
    embeddings: np.ndarray,
    labels: list[str],
    languages: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = np.asarray(labels)
    validation_folds = [validation for _, validation in folds]
    for c_value in config["ordinal_logistic"]["c_values"]:
        probabilities = np.zeros((len(labels), 3), dtype=float)
        started = time.perf_counter()
        for train, validation in folds:
            estimator = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=1000,
                random_state=config["random_seed"],
                solver="lbfgs",
            )
            estimator.fit(embeddings[train], y[train])
            probabilities[validation] = _aligned_probabilities(estimator, embeddings[validation])
        predictions = threshold_predictions(probabilities, "argmax")
        rows.append(
            _candidate_row(
                name=f"{model_spec['name']}__multiclass_balanced_C_{float(c_value):g}",
                model_type="jina_embedding_multiclass_logistic",
                model_id=model_spec["model_id"],
                revision=model_spec["revision"],
                c_value=float(c_value),
                decision_rule="argmax",
                labels=labels,
                languages=languages,
                predictions=predictions,
                validation_folds=validation_folds,
                parameters={"class_weight": "balanced"},
                runtime={**runtime, "fit_seconds": time.perf_counter() - started},
            )
        )
    return rows


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
            metrics = _with_ordinal_diagnostics(
                classification_metrics(labels, predictions), labels, predictions
            )
            candidates.append((lower_threshold, upper_threshold, metrics))
    eligible = [
        item
        for item in candidates
        if item[2]["per_class"]["Negative"]["precision"] >= 0.60
        and item[2]["per_class"]["Negative"]["recall"]
        >= baseline["per_class"]["Negative"]["recall"]
        and item[2]["macro_f1"] >= baseline["macro_f1"] - 0.01
    ]
    if eligible:
        return max(
            eligible,
            key=lambda item: (
                item[2]["per_class"]["Negative"]["recall"],
                item[2]["macro_f1"],
                -item[2]["ordinal_mae"],
                -item[2]["severe_error_rate"],
                item[2]["quadratic_weighted_kappa"],
            ),
        )
    return max(candidates, key=lambda item: item[2]["macro_f1"])


def _cross_fitted_threshold_predictions(
    labels: list[str],
    above_negative: np.ndarray,
    above_average: np.ndarray,
    baseline_predictions: list[str],
    fold_assignments: np.ndarray,
    threshold_values: list[float],
) -> tuple[list[str], list[dict[str, float]]]:
    predictions = np.empty(len(labels), dtype=object)
    threshold_evidence: list[dict[str, float]] = []
    for fold in sorted(set(fold_assignments.tolist())):
        tuning_indices = np.flatnonzero(fold_assignments != fold)
        evaluation_indices = np.flatnonzero(fold_assignments == fold)
        tuning_labels = [labels[index] for index in tuning_indices]
        tuning_baseline = classification_metrics(
            tuning_labels,
            [baseline_predictions[index] for index in tuning_indices],
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


def _embedding_ordinal_rows(
    model_spec: dict[str, Any],
    embeddings: np.ndarray,
    labels: list[str],
    languages: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    runtime: dict[str, Any],
    baseline_predictions: list[str],
) -> list[dict[str, Any]]:
    y = np.asarray(labels)
    settings = config["ordinal_logistic"]
    c_values = [float(value) for value in settings["c_values"]]
    threshold_values = [float(value) for value in settings["threshold_values"]]
    calibration_folds = int(settings["calibration_folds"])
    validation_folds = [validation for _, validation in folds]
    fold_assignments = np.full(len(labels), -1, dtype=int)
    baseline_metrics = classification_metrics(labels, baseline_predictions)
    rows: list[dict[str, Any]] = []

    for c_value in c_values:
        above_negative = np.zeros(len(labels), dtype=float)
        above_average = np.zeros(len(labels), dtype=float)
        started = time.perf_counter()
        for fold_number, (train, validation) in enumerate(folds):
            fold_assignments[validation] = fold_number
            lower_targets = (y[train] != "Negative").astype(int)
            upper_targets = (y[train] == "Positive").astype(int)
            lower_classifier = _calibrated_boundary_classifier(
                c_value=c_value,
                seed=config["random_seed"],
                calibration_folds=calibration_folds,
            )
            upper_classifier = _calibrated_boundary_classifier(
                c_value=c_value,
                seed=config["random_seed"],
                calibration_folds=calibration_folds,
            )
            lower_classifier.fit(embeddings[train], lower_targets)
            upper_classifier.fit(embeddings[train], upper_targets)
            above_negative[validation] = lower_classifier.predict_proba(embeddings[validation])[
                :, 1
            ]
            above_average[validation] = upper_classifier.predict_proba(embeddings[validation])[:, 1]
        fit_seconds = time.perf_counter() - started
        lower, upper, probability_evidence = project_monotonic_boundaries(
            above_negative, above_average
        )
        rows.append(
            _candidate_row(
                name=f"{model_spec['name']}__ordinal_composed_argmax_C_{c_value:g}",
                model_type="jina_embedding_two_boundary_ordinal_logistic",
                model_id=model_spec["model_id"],
                revision=model_spec["revision"],
                c_value=c_value,
                decision_rule="composed_probability_argmax",
                labels=labels,
                languages=languages,
                predictions=probability_argmax_labels(compose_ordinal_probabilities(lower, upper)),
                validation_folds=validation_folds,
                parameters={"class_weight": "balanced", "calibration": "sigmoid"},
                runtime={**runtime, "fit_seconds": fit_seconds},
                probability_evidence=probability_evidence,
            )
        )
        deployment_lower_threshold, deployment_upper_threshold, _ = _select_thresholds(
            labels,
            lower,
            upper,
            baseline_metrics,
            threshold_values,
        )
        threshold_predictions_by_fold, thresholds_by_fold = _cross_fitted_threshold_predictions(
            labels,
            lower,
            upper,
            baseline_predictions,
            fold_assignments,
            threshold_values,
        )
        rows.append(
            _candidate_row(
                name=f"{model_spec['name']}__ordinal_crossfit_threshold_C_{c_value:g}",
                model_type="jina_embedding_two_boundary_ordinal_logistic",
                model_id=model_spec["model_id"],
                revision=model_spec["revision"],
                c_value=c_value,
                decision_rule="cross_fitted_boundary_thresholds",
                labels=labels,
                languages=languages,
                predictions=threshold_predictions_by_fold,
                validation_folds=validation_folds,
                parameters={
                    "class_weight": "balanced",
                    "calibration": "sigmoid",
                    "deployment_lower_threshold": deployment_lower_threshold,
                    "deployment_upper_threshold": deployment_upper_threshold,
                },
                runtime={**runtime, "fit_seconds": fit_seconds},
                probability_evidence={
                    **probability_evidence,
                    "thresholds_by_fold": thresholds_by_fold,
                    "ordered_labels": ORDERED_LABELS,
                },
            )
        )
    return rows


def _gate_candidate(
    candidate: dict[str, Any], baseline: dict[str, Any], gates: dict[str, float]
) -> dict[str, bool]:
    return {
        "macro_f1": candidate["cv_macro_f1_mean"]
        >= baseline["cv_macro_f1_mean"] + gates["minimum_macro_f1_improvement"],
        "negative_precision": candidate["negative_precision"]
        >= gates["minimum_negative_precision"],
        "negative_recall": candidate["negative_recall"]
        >= baseline["negative_recall"] + gates["minimum_negative_recall_improvement"],
        "accuracy": candidate["oof_accuracy"]
        >= baseline["oof_accuracy"] - gates["maximum_accuracy_drop"],
        "stability": candidate["cv_macro_f1_std"]
        <= baseline["cv_macro_f1_std"] + gates["maximum_cv_macro_f1_std_increase"],
    }


def _select(rows: list[dict[str, Any]], baseline: dict[str, Any], gates: dict[str, float]):
    experimental_rows = [row for row in rows if row["model_type"] != "tfidf_word_char_logreg"]
    evaluated = [(row, _gate_candidate(row, baseline, gates)) for row in experimental_rows]
    eligible = [item for item in evaluated if all(item[1].values())]
    selected, checks = max(
        eligible or evaluated,
        key=lambda item: (
            item[0]["cv_macro_f1_mean"],
            item[0]["negative_f1"],
            item[0]["quadratic_weighted_kappa"],
            -item[0]["severe_error_rate"],
        ),
    )
    return selected, checks, bool(eligible)


def _write_report(
    frame: pd.DataFrame,
    baseline: dict[str, Any],
    selected: dict[str, Any],
    checks: dict[str, bool],
    decision: dict[str, Any],
    path: Path,
) -> None:
    columns = [
        "name",
        "model_type",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "oof_accuracy",
        "negative_precision",
        "negative_recall",
        "negative_f1",
        "ordinal_mae",
        "quadratic_weighted_kappa",
        "severe_error_rate",
    ]
    table = frame.sort_values("cv_macro_f1_mean", ascending=False).head(12)[columns].copy()
    for column in columns[2:]:
        table[column] = table[column].round(4)
    gate_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} - `{name}`" for name, passed in checks.items()
    )
    content = f"""# Jina v3 ordinal-logistic experiment

## Decision

Selected OOF candidate: `{selected["name"]}`.

CV Macro-F1 is {selected["cv_macro_f1_mean"]:.4f}, versus {baseline["cv_macro_f1_mean"]:.4f} for the official TF-IDF baseline. Negative precision/recall/F1 are {selected["negative_precision"]:.4f}/{selected["negative_recall"]:.4f}/{selected["negative_f1"]:.4f}.

Production model and held-out test were not changed. This is a Colab/GPU research experiment using frozen Jina embeddings plus ordinal boundary classifiers.

## Promotion gates

{gate_lines}

Metric gates passed: `{decision["metric_gates_passed"]}`.

## Method

- Recreated and hash-verified the frozen 3,838-row training split and 960-row reserved holdout split.
- Encoded Dutch and English training reviews together with revision-pinned Jina v3 classification embeddings.
- Compared balanced multiclass Logistic Regression against two calibrated ordinal boundaries: `Negative < Average < Positive`.
- Selected all C values and boundary thresholds only from training-set out-of-fold predictions.
- Did not evaluate the reserved holdout rows, replace `artifacts/model.joblib`, or train a separate English model.

## Top OOF configurations

{table.to_markdown(index=False)}

## Limitations

Jina Embeddings v3 is CC-BY-NC-4.0, so this branch is non-commercial research evidence. English slice metrics remain directional because English Negative support is tiny. A final promotion decision still needs a newly collected blind test.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    from .config import load_config

    training = load_config(config["training_config"])
    seed = int(training["random_seed"])
    config["random_seed"] = seed
    raw = load_dataset(training["data_path"])
    annotated, _ = annotate_review_languages(raw, DutchLanguageDetector(**training["language"]))
    split = make_holdout_split(
        annotated,
        test_size=float(training["test_size"]),
        random_seed=seed,
        stratify_columns=("detected_language", "Label"),
    )
    train_hash = _hash_values(split.train["normalized_review"].tolist())
    test_hash = _hash_values(split.test["normalized_review"].tolist())
    if train_hash != config["expected_train_normalized_sha256"]:
        raise RuntimeError("Frozen training split hash changed; refusing to run")
    if test_hash != config["expected_test_normalized_sha256"]:
        raise RuntimeError("Frozen held-out split hash changed; refusing to run")

    reviews = split.train["Reviews"].astype(str).tolist()
    labels = split.train["Label"].astype(str).tolist()
    languages = split.train["detected_language"].astype(str).tolist()
    strata = split.train[["detected_language", "Label"]].astype(str).agg("::".join, axis=1)
    splitter = StratifiedKFold(n_splits=int(training["cv_folds"]), shuffle=True, random_state=seed)
    folds = list(splitter.split(reviews, strata))
    baseline = _official_baseline(
        reviews,
        labels,
        languages,
        folds,
        {**training["model"], "random_seed": seed},
    )
    rows = [baseline]
    for model_spec in config["models"]:
        embeddings, runtime = _encode_or_load(model_spec, reviews, config)
        rows.extend(
            _embedding_multiclass_rows(
                model_spec,
                embeddings,
                labels,
                languages,
                folds,
                config,
                runtime,
            )
        )
        multiclass_baseline = max(
            [row for row in rows if row["model_type"] == "jina_embedding_multiclass_logistic"],
            key=lambda row: row["cv_macro_f1_mean"],
        )
        baseline_predictions = threshold_predictions(
            _oof_multiclass_probabilities(
                embeddings,
                labels,
                folds,
                float(multiclass_baseline["C"]),
                seed,
            ),
            "argmax",
        )
        rows.extend(
            _embedding_ordinal_rows(
                model_spec,
                embeddings,
                labels,
                languages,
                folds,
                config,
                runtime,
                baseline_predictions,
            )
        )

    selected, checks, passed = _select(rows, baseline, config["promotion_gates"])
    selected_model = next(
        spec for spec in config["models"] if spec["model_id"] == selected["model_id"]
    )
    decision = {
        "status": "training_oof_only",
        "selected_candidate": selected["name"],
        "selected_metrics": {
            key: selected[key]
            for key in (
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
                "oof_accuracy",
                "negative_precision",
                "negative_recall",
                "negative_f1",
                "ordinal_mae",
                "quadratic_weighted_kappa",
                "severe_error_rate",
                "dutch_macro_f1",
                "english_macro_f1",
            )
            if key in selected
        },
        "gate_checks": checks,
        "metric_gates_passed": passed,
        "production_promotion_eligible": False,
        "official_model_replaced": False,
        "heldout_evaluated": False,
        "license": selected_model["license"],
        "provenance": {
            "embedding_model": selected_model["model_id"],
            "embedding_revision": selected_model["revision"],
            "remote_code_dependency": selected_model.get("remote_code_dependency"),
            "remote_code_revision": selected_model.get("remote_code_revision"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
        },
        "data": {
            "raw_sha256": sha256_file(training["data_path"]),
            "train_rows": len(split.train),
            "heldout_rows": len(split.test),
            "train_normalized_sha256": train_hash,
            "heldout_normalized_sha256": test_hash,
        },
    }
    frame = pd.DataFrame(rows)
    output = Path(config["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    decision_path = Path(config["decision_json"])
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(frame, baseline, selected, checks, decision, Path(config["report_path"]))
    return decision


def _oof_multiclass_probabilities(
    embeddings: np.ndarray,
    labels: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_value: float,
    seed: int,
) -> np.ndarray:
    probabilities = np.zeros((len(labels), 3), dtype=float)
    y = np.asarray(labels)
    for train, validation in folds:
        estimator = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
        estimator.fit(embeddings[train], y[train])
        probabilities[validation] = _aligned_probabilities(estimator, embeddings[validation])
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/jina_ordinal_logistic.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    print(json.dumps(run_experiment(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
