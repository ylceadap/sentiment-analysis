from pathlib import Path

import pandas as pd
import pytest

from dutch_sentiment.data import deduplicate_reviews, load_dataset, make_holdout_split


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    pd.DataFrame(rows, columns=["Reviews", "Label"]).to_csv(path, index=False, encoding="utf-8-sig")


def test_load_dataset_validates_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["goed"], "Label": ["Positive"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Expected columns"):
        load_dataset(path)


def test_load_dataset_rejects_missing_and_unexpected_labels(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    _write_csv(missing, [("goed", "Positive"), (None, "Average")])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="missing"):
        load_dataset(missing)

    labels = tmp_path / "labels.csv"
    _write_csv(labels, [("goed", "Unknown")])
    with pytest.raises(ValueError, match="Unexpected labels"):
        load_dataset(labels)


def test_deduplication_removes_same_label_and_conflicting_groups() -> None:
    frame = pd.DataFrame(
        {
            "Reviews": ["Heel goed", "Heel\u200b goed", "Matig", "Matig", "Slecht"],
            "Label": ["Positive", "Positive", "Average", "Negative", "Negative"],
        }
    )
    clean, duplicate_rows, conflicting_groups = deduplicate_reviews(frame)
    assert clean["Reviews"].tolist() == ["Heel goed", "Slecht"]
    assert duplicate_rows == 1
    assert conflicting_groups == 1


def test_holdout_is_deterministic_stratified_and_has_no_normalized_overlap() -> None:
    rows = []
    for label in ("Positive", "Average", "Negative"):
        rows.extend((f"{label} unieke review {index}", label) for index in range(20))
    frame = pd.DataFrame(rows, columns=["Reviews", "Label"])
    first = make_holdout_split(frame, test_size=0.25, random_seed=7)
    second = make_holdout_split(frame, test_size=0.25, random_seed=7)
    assert first.test["Reviews"].tolist() == second.test["Reviews"].tolist()
    assert not set(first.train["normalized_review"]) & set(first.test["normalized_review"])
    assert first.test["Label"].value_counts().max() - first.test["Label"].value_counts().min() <= 1
