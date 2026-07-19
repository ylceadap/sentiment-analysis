from __future__ import annotations

from pathlib import Path

from dutch_sentiment.benchmark import _measure, run_benchmark
from dutch_sentiment.reporting import build_model_report
from dutch_sentiment.train import _experiment_specs, _git_commit, _git_dirty, _metrics_by_language


def test_reporting_renders_complete_current_evidence(tmp_path: Path) -> None:
    """The report builder consumes current artifacts and writes all major sections."""
    output = tmp_path / "model-report.md"
    build_model_report("artifacts", output)
    content = output.read_text()
    assert "# Model Report" in content
    assert "Held-out" in content
    assert "latency" in content


def test_measure_and_real_benchmark_contract() -> None:
    """Timing helpers and the real artifact return the documented benchmark schema."""
    measured = _measure(lambda text: len(text), iterations=3, warmup=1)
    assert measured["iterations"] == 3
    assert measured["max_ms"] >= measured["p50_ms"]
    result = run_benchmark("artifacts/model.joblib", iterations=2, warmup=0)
    assert result["model_artifact_bytes"] > 0
    assert result["model_sha256"]
    assert result["measurements"]["http_classify"]["iterations"] == 30


def test_training_helpers_keep_candidate_and_language_contracts() -> None:
    """Training helpers expose bounded candidates, Git state, and language slices."""
    assert len(_experiment_specs()) == 6
    assert _git_commit()
    assert isinstance(_git_dirty(), bool)
    metrics = _metrics_by_language(
        ["dutch", "dutch", "dutch", "english", "english", "english"],
        ["Positive", "Average", "Negative", "Positive", "Average", "Negative"],
        ["Positive", "Average", "Negative", "Positive", "Average", "Average"],
        [
            {"Positive": 0.8, "Average": 0.1, "Negative": 0.1},
            {"Positive": 0.1, "Average": 0.8, "Negative": 0.1},
            {"Positive": 0.1, "Average": 0.1, "Negative": 0.8},
            {"Positive": 0.8, "Average": 0.1, "Negative": 0.1},
            {"Positive": 0.1, "Average": 0.8, "Negative": 0.1},
            {"Positive": 0.1, "Average": 0.8, "Negative": 0.1},
        ],
    )
    assert set(metrics) == {"dutch", "english"}
