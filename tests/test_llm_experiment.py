from __future__ import annotations

import json

import pandas as pd
import pytest

from dutch_sentiment.llm_experiment import (
    _build_system_prompt,
    _cache_path,
    _cost_summary,
    _load_api_key,
    _parse_label,
    _promotion_checks,
    _select_shots,
)


def test_loads_api_key_from_ignored_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    key_file = tmp_path / "deepseek_api_key"
    key_file.write_text("secret-value\n", encoding="utf-8")
    assert (
        _load_api_key({"api_key_env": "DEEPSEEK_API_KEY", "api_key_file": str(key_file)})
        == "secret-value"
    )


def _shot_frame() -> pd.DataFrame:
    rows = []
    source = 2
    for label in ("Positive", "Average", "Negative"):
        for language, count in (("dutch", 8), ("english", 4)):
            for number in range(count):
                rows.append(
                    {
                        "Reviews": f"{language} {label} example {number} " + "x" * (300 + number),
                        "Label": label,
                        "detected_language": language,
                        "source_row": source,
                    }
                )
                source += 1
    return pd.DataFrame(rows)


def test_selects_exactly_24_balanced_shots() -> None:
    shots = _select_shots(
        _shot_frame(),
        {
            "shot_language_targets": {"dutch": 6, "english": 2},
            "shots_per_label": 8,
            "shot_max_characters": 900,
        },
    )
    assert len(shots) == 24
    counts: dict[tuple[str, str], int] = {}
    for shot in shots:
        counts[(shot.label, shot.language)] = counts.get((shot.label, shot.language), 0) + 1
    assert set(counts.values()) == {2, 6}
    assert "24 labeled examples" in _build_system_prompt(shots)


def test_json_label_parser_is_strict() -> None:
    assert _parse_label('{"label":"Average"}') == "Average"
    with pytest.raises(ValueError):
        _parse_label("Positive")
    with pytest.raises(ValueError):
        _parse_label(json.dumps({"label": "Neutral"}))
    with pytest.raises(ValueError):
        _parse_label(json.dumps({"label": "Positive", "reason": "good"}))


def test_cache_key_changes_with_model_prompt_or_review(tmp_path) -> None:
    original = _cache_path(tmp_path, "model-a", "prompt-a", "review-a")
    assert original != _cache_path(tmp_path, "model-b", "prompt-a", "review-a")
    assert original != _cache_path(tmp_path, "model-a", "prompt-b", "review-a")
    assert original != _cache_path(tmp_path, "model-a", "prompt-a", "review-b")


def test_promotion_requires_all_predeclared_gates() -> None:
    baseline = {"macro_f1": 0.64, "accuracy": 0.65}
    heldout = {
        "macro_f1": 0.67,
        "accuracy": 0.65,
        "invalid_response_rate": 0.0,
        "per_class": {"Negative": {"precision": 0.65, "recall": 0.60}},
    }
    gates = {
        "minimum_heldout_macro_f1_improvement": 0.015,
        "minimum_negative_precision": 0.60,
        "minimum_negative_recall": 0.52,
        "maximum_invalid_response_rate": 0.005,
        "maximum_heldout_accuracy_drop": 0.01,
    }
    assert all(_promotion_checks(heldout, baseline, gates).values())


def test_cost_summary_does_not_report_legacy_queue_time_as_latency() -> None:
    frame = pd.DataFrame(
        [
            {
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 105,
                "cache_hit": False,
                "latency_ms": 120_000,
            }
        ]
    )
    summary = _cost_summary(frame)
    assert summary["latency_measurement"] == "unavailable_for_legacy_run"
    assert summary["latency_p50_ms"] is None
