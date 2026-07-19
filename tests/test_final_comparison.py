from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from dutch_sentiment import final_comparison


def test_restore_loaded_logistic_state_repairs_legacy_multiclass_head() -> None:
    """A loaded multiclass head remains usable when sklearn omitted retired state."""
    classifier = LogisticRegression()
    classifier.classes_ = pd.Series(["Average", "Negative", "Positive"]).to_numpy()

    restored = final_comparison._restore_loaded_logistic_state(classifier)

    assert restored is classifier
    assert classifier.multi_class == "multinomial"


def test_deepseek_predictions_require_source_row_hash_and_actual_label(monkeypatch) -> None:
    """Archived API labels are accepted only after independent row-identity checks."""
    review = "  Goede\nfilm  "
    normalized = "Goede film"
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    csv = (
        "split,source_row,status,review_sha256,actual,label\n"
        f"heldout,9,ok,{digest},Positive,Average\n"
    ).encode()
    monkeypatch.setattr(final_comparison, "_git_bytes", lambda *args: csv)
    heldout = pd.DataFrame([{"source_row": 9, "Reviews": review, "Label": "Positive"}])

    assert final_comparison._deepseek_predictions(
        heldout,
        {"deepseek_archive": "archive", "deepseek_predictions": "predictions.csv"},
    ) == ["Average"]


def test_common_metrics_include_ordinal_and_language_evidence() -> None:
    """Every final candidate receives the same classification and ordinal metrics."""
    metrics = final_comparison._metrics(
        ["Positive", "Average", "Negative"] * 2,
        ["Positive", "Negative", "Negative"] * 2,
        ["dutch"] * 3 + ["english"] * 3,
    )

    assert metrics["macro_f1"] > 0
    assert metrics["ordinal_mae"] == 1 / 3
    assert set(metrics["by_language"]) == {"dutch", "english"}


def test_report_discloses_reused_holdout(tmp_path: Path) -> None:
    """The presentation report cannot imply that reused evidence is a new blind test."""
    summary = pd.DataFrame([{"rank": 1, "model": "demo", "macro_f1": 0.5}])
    path = tmp_path / "report.md"
    final_comparison._write_report(summary, {"data": {"heldout_rows": 3}}, path)

    content = path.read_text()
    assert "reused-heldout presentation comparison" in content
    assert "not a new blind test" in content
    assert "post_evaluation_tuning_allowed=false" in content
