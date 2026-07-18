"""Validated dataset loading, deduplication, and leakage-safe splitting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import EXPECTED_COLUMNS, LABELS
from .text import normalize_text


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest without modifying the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the UTF-8 BOM CSV and enforce its public schema and label domain."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    frame = pd.read_csv(data_path, encoding="utf-8-sig")
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected columns {list(EXPECTED_COLUMNS)}, found {frame.columns.tolist()}"
        )
    if frame[list(EXPECTED_COLUMNS)].isna().any().any():
        missing = frame[list(EXPECTED_COLUMNS)].isna().sum().to_dict()
        raise ValueError(f"Dataset contains missing required values: {missing}")
    unexpected = sorted(set(frame["Label"]) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")
    if frame["Reviews"].astype(str).str.strip().eq("").any():
        raise ValueError("Dataset contains empty or whitespace-only reviews")
    return frame


@dataclass(frozen=True)
class PreparedSplit:
    """Leakage-safe holdout with metadata needed for traceability."""

    train: pd.DataFrame
    test: pd.DataFrame
    duplicate_rows_removed: int
    conflicting_groups_removed: int


def deduplicate_reviews(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Remove conflicting normalized groups, then keep one row per normalized review."""
    work = frame.copy()
    work["normalized_review"] = work["Reviews"].map(normalize_text)
    label_counts = work.groupby("normalized_review")["Label"].transform("nunique")
    conflicting_groups = int(work.loc[label_counts.gt(1), "normalized_review"].nunique())
    without_conflicts = work.loc[label_counts.eq(1)].copy()
    before = len(without_conflicts)
    deduplicated = without_conflicts.drop_duplicates("normalized_review", keep="first")
    return deduplicated.reset_index(drop=True), before - len(deduplicated), conflicting_groups


def make_holdout_split(
    frame: pd.DataFrame, *, test_size: float = 0.2, random_seed: int = 42
) -> PreparedSplit:
    """Create a deterministic shuffled stratified split after normalized deduplication."""
    clean, duplicate_rows, conflicting_groups = deduplicate_reviews(frame)
    train, test = train_test_split(
        clean,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=clean["Label"],
    )
    overlap = set(train["normalized_review"]) & set(test["normalized_review"])
    if overlap:
        raise RuntimeError("Normalized duplicate leakage detected across holdout split")
    return PreparedSplit(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        duplicate_rows_removed=duplicate_rows,
        conflicting_groups_removed=conflicting_groups,
    )
