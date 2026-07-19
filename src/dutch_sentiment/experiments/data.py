"""Prepare the shared frozen train/holdout split used by research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..config import load_config
from ..data import annotate_review_languages, load_dataset, make_holdout_split, sha256_file
from ..language import DutchLanguageDetector
from .common import hash_values


@dataclass(frozen=True)
class FrozenExperimentData:
    """Contain hash-verified training rows, fixed folds, and provenance metadata."""

    reviews: list[str]
    labels: list[str]
    languages: list[str]
    folds: list[tuple[np.ndarray, np.ndarray]]
    training_config: dict[str, Any]
    seed: int
    raw_sha256: str
    train_sha256: str
    heldout_sha256: str
    train_rows: int
    heldout_rows: int


def prepare_frozen_experiment(config: dict[str, Any]) -> FrozenExperimentData:
    """Recreate and verify the immutable split without evaluating held-out labels."""
    training = load_config(config["training_config"])
    seed = int(training["random_seed"])
    raw = load_dataset(training["data_path"])
    detector = DutchLanguageDetector(**training["language"])
    annotated, _ = annotate_review_languages(raw, detector)
    split = make_holdout_split(
        annotated,
        test_size=float(training["test_size"]),
        random_seed=seed,
        stratify_columns=("detected_language", "Label"),
    )
    train_hash = hash_values(split.train["normalized_review"].tolist())
    heldout_hash = hash_values(split.test["normalized_review"].tolist())
    if train_hash != config["expected_train_normalized_sha256"]:
        raise RuntimeError("Frozen training split hash changed; refusing to run")
    if heldout_hash != config["expected_test_normalized_sha256"]:
        raise RuntimeError("Frozen held-out split hash changed; refusing to run")

    reviews = split.train["Reviews"].astype(str).tolist()
    labels = split.train["Label"].astype(str).tolist()
    languages = split.train["detected_language"].astype(str).tolist()
    strata = split.train[["detected_language", "Label"]].astype(str).agg("::".join, axis=1)
    splitter = StratifiedKFold(n_splits=int(training["cv_folds"]), shuffle=True, random_state=seed)
    return FrozenExperimentData(
        reviews=reviews,
        labels=labels,
        languages=languages,
        folds=list(splitter.split(reviews, strata)),
        training_config=training,
        seed=seed,
        raw_sha256=sha256_file(training["data_path"]),
        train_sha256=train_hash,
        heldout_sha256=heldout_hash,
        train_rows=len(split.train),
        heldout_rows=len(split.test),
    )
