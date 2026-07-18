"""Train-only OOF comparison of frozen sentence embeddings and the official baseline."""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .config import load_config
from .constants import LABELS
from .data import annotate_review_languages, load_dataset, make_holdout_split, sha256_file
from .language import DutchLanguageDetector
from .metrics import classification_metrics
from .model import ModelSpec, build_pipeline

LOGGER = logging.getLogger(__name__)


def _hash_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _hash_reviews(values: list[str]) -> str:
    raw = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _aligned_probabilities(estimator: Any, features: Any) -> np.ndarray:
    raw = estimator.predict_proba(features)
    columns = {str(label): raw[:, i] for i, label in enumerate(estimator.classes_)}
    aligned = np.column_stack([columns[label] for label in LABELS])
    return aligned / aligned.sum(axis=1, keepdims=True)


def threshold_predictions(probabilities: np.ndarray, threshold: str | float) -> list[str]:
    """Apply a Negative one-vs-rest threshold, otherwise use standard argmax."""
    if threshold == "argmax":
        return [LABELS[index] for index in probabilities.argmax(axis=1)]
    value = float(threshold)
    if not 0 < value < 1:
        raise ValueError("Negative threshold must be between zero and one")
    negative = LABELS.index("Negative")
    alternatives = [index for index, label in enumerate(LABELS) if label != "Negative"]
    return [
        "Negative" if row[negative] >= value else LABELS[max(alternatives, key=row.__getitem__)]
        for row in probabilities
    ]


def _cache_path(
    cache_dir: Path, model_name: str, revision: str, review_hash: str, normalized: bool
) -> Path:
    key = hashlib.sha256(f"{model_name}|{revision}|{review_hash}|{normalized}".encode()).hexdigest()
    return cache_dir / f"{model_name}-{key[:16]}.npz"


def _encode_or_load(
    model_spec: dict[str, Any], reviews: list[str], config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    review_hash = _hash_reviews(reviews)
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = bool(config["normalize_embeddings"])
    cache = _cache_path(
        cache_dir, model_spec["name"], model_spec["revision"], review_hash, normalized
    )
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored["review_hash"].item()) != review_hash:
                raise RuntimeError(f"Embedding cache hash mismatch: {cache}")
            embeddings = stored["embeddings"].astype(np.float32, copy=False)
        return embeddings, {
            "cache_hit": True,
            "cache_path": str(cache),
            "cache_sha256": sha256_file(cache),
            "encode_seconds": 0.0,
            "embedding_dimension": embeddings.shape[1],
        }
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Run `make install-embeddings` before this experiment") from exc
    started = time.perf_counter()
    model = SentenceTransformer(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_folder=config["huggingface_cache_dir"],
        device="cpu",
    )
    embeddings = model.encode(
        reviews,
        batch_size=int(config["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalized,
    ).astype(np.float32, copy=False)
    np.savez_compressed(
        cache,
        embeddings=embeddings,
        review_hash=np.asarray(review_hash),
        model_id=np.asarray(model_spec["model_id"]),
        revision=np.asarray(model_spec["revision"]),
        normalized=np.asarray(normalized),
    )
    return embeddings, {
        "cache_hit": False,
        "cache_path": str(cache),
        "cache_sha256": sha256_file(cache),
        "encode_seconds": time.perf_counter() - started,
        "embedding_dimension": embeddings.shape[1],
        "max_sequence_length": int(model.max_seq_length),
    }


def _metric_row(
    *,
    name: str,
    model_type: str,
    model_id: str,
    revision: str,
    class_weight: str,
    c_value: float,
    threshold: str | float,
    labels: list[str],
    languages: list[str],
    predictions: list[str],
    probabilities: np.ndarray,
    validation_folds: list[np.ndarray],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    probability_rows = [
        {label: float(row[index]) for index, label in enumerate(LABELS)} for row in probabilities
    ]
    metrics = classification_metrics(labels, predictions, probability_rows)
    fold_metrics = [
        classification_metrics(
            [labels[index] for index in fold], [predictions[index] for index in fold]
        )
        for fold in validation_folds
    ]
    result: dict[str, Any] = {
        "name": name,
        "model_type": model_type,
        "model_id": model_id,
        "revision": revision,
        "class_weight": class_weight,
        "C": c_value,
        "negative_threshold": str(threshold),
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
        "oof_macro_f1": metrics["macro_f1"],
        "oof_balanced_accuracy": metrics["balanced_accuracy"],
        "oof_accuracy": metrics["accuracy"],
        "negative_precision": metrics["per_class"]["Negative"]["precision"],
        "negative_recall": metrics["per_class"]["Negative"]["recall"],
        "negative_f1": metrics["per_class"]["Negative"]["f1-score"],
        "per_class_json": json.dumps(metrics["per_class"], sort_keys=True),
        "confusion_matrix_json": json.dumps(metrics["confusion_matrix"]),
        **runtime,
    }
    for language in sorted(set(languages)):
        indices = [index for index, value in enumerate(languages) if value == language]
        sliced = classification_metrics(
            [labels[index] for index in indices], [predictions[index] for index in indices]
        )
        result[f"{language}_rows"] = len(indices)
        result[f"{language}_macro_f1"] = sliced["macro_f1"]
        result[f"{language}_accuracy"] = sliced["accuracy"]
        result[f"{language}_negative_recall"] = sliced["per_class"]["Negative"]["recall"]
    return result


def _official_baseline(
    reviews: list[str],
    labels: list[str],
    languages: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    probabilities = np.zeros((len(labels), len(LABELS)))
    started = time.perf_counter()
    for train, validation in folds:
        pipeline = build_pipeline(
            ModelSpec("combined_balanced_ratings", "combined", "balanced"), model_config
        )
        pipeline.fit([reviews[i] for i in train], [labels[i] for i in train])
        probabilities[validation] = _aligned_probabilities(
            pipeline, [reviews[i] for i in validation]
        )
    predictions = threshold_predictions(probabilities, "argmax")
    return _metric_row(
        name="official_tfidf_baseline",
        model_type="tfidf_word_char_logreg",
        model_id="official_combined_balanced_ratings",
        revision="submission-v1",
        class_weight="balanced",
        c_value=1.0,
        threshold="argmax",
        labels=labels,
        languages=languages,
        predictions=predictions,
        probabilities=probabilities,
        validation_folds=[validation for _, validation in folds],
        runtime={"fit_seconds": time.perf_counter() - started},
    )


def _embedding_rows(
    model_spec: dict[str, Any],
    embeddings: np.ndarray,
    labels: list[str],
    languages: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    settings = config["logistic_regression"]
    y = np.asarray(labels)
    for c_value in settings["c_values"]:
        for weight in settings["class_weights"]:
            probabilities = np.zeros((len(labels), len(LABELS)))
            started = time.perf_counter()
            for train, validation in folds:
                estimator = LogisticRegression(
                    C=float(c_value),
                    class_weight=weight["value"],
                    max_iter=1000,
                    random_state=config["random_seed"],
                    solver="lbfgs",
                )
                estimator.fit(embeddings[train], y[train])
                probabilities[validation] = _aligned_probabilities(
                    estimator, embeddings[validation]
                )
            fit_seconds = time.perf_counter() - started
            for threshold in settings["negative_thresholds"]:
                rows.append(
                    _metric_row(
                        name=(
                            f"{model_spec['name']}__c{c_value}__{weight['name']}"
                            f"__threshold_{threshold}"
                        ),
                        model_type="frozen_sentence_embedding_logreg",
                        model_id=model_spec["model_id"],
                        revision=model_spec["revision"],
                        class_weight=weight["name"],
                        c_value=float(c_value),
                        threshold=threshold,
                        labels=labels,
                        languages=languages,
                        predictions=threshold_predictions(probabilities, threshold),
                        probabilities=probabilities,
                        validation_folds=[validation for _, validation in folds],
                        runtime={**runtime, "fit_seconds": fit_seconds},
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


def _select(
    rows: list[dict[str, Any]], baseline: dict[str, Any], gates: dict[str, float]
) -> tuple[dict[str, Any], dict[str, bool], bool]:
    evaluated = [
        (row, _gate_candidate(row, baseline, gates))
        for row in rows
        if row["model_type"] == "frozen_sentence_embedding_logreg"
    ]
    eligible = [item for item in evaluated if all(item[1].values())]
    selected, checks = max(
        eligible or evaluated,
        key=lambda item: (
            item[0]["cv_macro_f1_mean"],
            item[0]["negative_recall"],
            -item[0]["cv_macro_f1_std"],
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
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "oof_accuracy",
        "negative_precision",
        "negative_recall",
        "negative_f1",
        "dutch_macro_f1",
        "english_macro_f1",
    ]
    table = frame.sort_values("cv_macro_f1_mean", ascending=False).head(10)[columns].copy()
    for column in columns[1:]:
        table[column] = table[column].round(4)
    gates = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in checks.items()
    )
    advance = decision["advance_to_gpu_finetuning"]
    outcome = (
        "Advance to a separately budgeted GPU fine-tune and a new blind test."
        if advance
        else "Stop: no frozen-embedding candidate cleared every predeclared OOF gate."
    )
    content = f"""# Frozen sentence-embedding experiment

## Decision

**{outcome}** The official TF-IDF model and held-out test were not changed or evaluated.

Best experimental candidate: `{selected["name"]}`. CV Macro-F1 is {selected["cv_macro_f1_mean"]:.4f}, versus {baseline["cv_macro_f1_mean"]:.4f} for the official baseline. Negative precision/recall are {selected["negative_precision"]:.4f}/{selected["negative_recall"]:.4f}, versus {baseline["negative_precision"]:.4f}/{baseline["negative_recall"]:.4f}.

## Promotion gates

{gates}

## Method

- Recreated and hash-verified the frozen 3,838-row training split and 960-row holdout split.
- Encoded Dutch and English training reviews together using two frozen, revision-pinned CPU encoders.
- Used the same five stratified folds for all candidates; labels never fitted the encoders.
- Selected C, class weight, and Negative threshold only from out-of-fold predictions.
- Did not compute holdout metrics, replace `artifacts/model.joblib`, or train an English-only model.

## Top OOF configurations

{table.to_markdown(index=False)}

## Reproducibility and limitations

Exact model revisions, split hashes, search grid, cache paths, and gates are in `configs/embedding_experiment.yaml`. Embeddings and model downloads are local-only. English slice results are directional: training has 388 English rows and only 8 English Negative rows. A future final decision requires a newly collected blind test.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
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
        reviews, labels, languages, folds, {**training["model"], "random_seed": seed}
    )
    rows = [baseline]
    for model_spec in config["models"]:
        embeddings, runtime = _encode_or_load(model_spec, reviews, config)
        rows.extend(
            _embedding_rows(model_spec, embeddings, labels, languages, folds, config, runtime)
        )
    selected, checks, advance = _select(rows, baseline, config["promotion_gates"])
    decision = {
        "advance_to_gpu_finetuning": advance,
        "selected_candidate": selected["name"],
        "gate_checks": checks,
        "official_model_replaced": False,
        "heldout_evaluated": False,
        "next_step": (
            "GPU fine-tune the winner, then evaluate once on a new blind test"
            if advance
            else "Collect and review more Dutch Negative examples before GPU fine-tuning"
        ),
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
    output = Path(config["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    decision_path = Path(config["decision_json"])
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(frame, baseline, selected, checks, decision, Path(config["report_path"]))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_experiment.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    print(json.dumps(run_experiment(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
