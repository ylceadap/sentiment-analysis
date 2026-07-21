from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dutch_sentiment.experiments.robbert_finetune import (
    _encode_batch,
    _loss_function,
    verify_bundle,
)
from dutch_sentiment.experiments.robbert_improvement import (
    _select_screened,
    write_trial_index,
)
from dutch_sentiment.experiments.robbert_improvement import (
    verify_bundle as verify_improvement_bundle,
)
from dutch_sentiment.models.robbert import (
    balanced_boundary_weights,
    balanced_class_weights,
    coral_probabilities,
    labels_from_probabilities,
    multiclass_targets,
    ordinal_targets,
    probability_rows,
    softmax_probabilities,
)


def test_robbert_targets_and_balancing_follow_project_label_contract() -> None:
    """Both objectives encode labels deterministically and derive finite train-only weights."""
    labels = ["Positive", "Average", "Negative", "Positive", "Average"]
    assert multiclass_targets(labels).tolist() == [0, 1, 2, 0, 1]
    assert ordinal_targets(labels).tolist() == [[1, 1], [1, 0], [0, 0], [1, 1], [1, 0]]
    assert np.all(np.isfinite(balanced_class_weights(labels)))
    assert balanced_boundary_weights(labels).shape == (2, 2)
    with pytest.raises(ValueError, match="Unknown sentiment"):
        ordinal_targets(["Unknown"])


def test_robbert_probability_adapters_preserve_order_and_monotonicity() -> None:
    """Multiclass and CORAL logits both produce valid project-ordered probability rows."""
    multiclass = softmax_probabilities(np.asarray([[4.0, 1.0, -2.0]]))
    assert labels_from_probabilities(multiclass) == ["Positive"]
    assert probability_rows(multiclass)[0]["Positive"] > 0.9

    # First boundary is easier than the second, so Average receives their difference.
    ordinal = coral_probabilities(np.asarray([[2.0, -1.0]]))
    assert np.allclose(ordinal.sum(axis=1), 1.0)
    assert probability_rows(ordinal)[0]["Average"] > 0
    with pytest.raises(ValueError, match="shape"):
        coral_probabilities(np.zeros((2, 3)))


class _TinyTokenizer:
    """Provide the tokenizer methods needed to test explicit truncation without downloads."""

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [[int(value) for value in text.split()] for text in texts]}

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [0, *token_ids, 2]

    def pad(self, rows: list[dict[str, list[int]]], **_: object) -> dict[str, object]:
        import torch

        width = max(len(row["input_ids"]) for row in rows)
        return {
            key: torch.tensor([row[key] + [0] * (width - len(row[key])) for row in rows])
            for key in ("input_ids", "attention_mask")
        }


def test_robbert_last_and_head_tail_strategies_keep_intended_tokens() -> None:
    """Long reviews retain their conclusion or both endpoints within the exact token budget."""
    tokenizer = _TinyTokenizer()
    text = "1 2 3 4 5 6 7 8 9 10"
    last = _encode_batch(tokenizer, [text], max_length=8, input_strategy="last")
    head_tail = _encode_batch(tokenizer, [text], max_length=8, input_strategy="head_tail")
    assert last["input_ids"].tolist() == [[0, 5, 6, 7, 8, 9, 10, 2]]
    assert head_tail["input_ids"].tolist() == [[0, 1, 2, 3, 8, 9, 10, 2]]


def test_robbert_improvement_losses_are_finite_and_screening_guards_average() -> None:
    """Alternative losses train safely and collapsed Average candidates cannot be promoted."""
    import torch

    labels = ["Positive", "Average", "Negative"] * 2
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.0]] * 2)
    targets = torch.tensor([0, 1, 2] * 2)
    for name in ("cross_entropy", "mild_class_weights", "focal", "ordinal_aware"):
        loss = _loss_function("multiclass_logistic", labels, "cpu", loss_name=name)
        assert torch.isfinite(loss(logits, targets))

    summaries = [
        {"candidate_id": "collapsed", "macro_f1_mean": 0.9, "average_recall_mean": 0.0},
        {"candidate_id": "sound", "macro_f1_mean": 0.7, "average_recall_mean": 0.8},
    ]
    assert _select_screened(summaries, 1, 0.4) == ["sound"]


def test_robbert_trial_index_summarizes_without_copying_predictions(tmp_path: Path) -> None:
    """The compact index exposes trial scores without duplicating row-level probabilities."""
    trial_path = tmp_path / "trials" / "candidate-a" / "fold-0-seed-42.json"
    trial_path.parent.mkdir(parents=True)
    trial_path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-a",
                "fold": 0,
                "seed": 42,
                "best_epoch": 3,
                "seconds": 12.5,
                "metrics": {
                    "macro_f1": 0.7,
                    "accuracy": 0.8,
                    "per_class": {"Average": {"recall": 0.6}},
                },
                "probabilities": [[0.1, 0.2, 0.7]],
            }
        ),
        encoding="utf-8",
    )

    index = write_trial_index(tmp_path)
    row = json.loads(index.read_text())

    assert row["candidate_id"] == "candidate-a"
    assert row["macro_f1"] == 0.7
    assert "probabilities" not in row


def test_robbert_improvement_evidence_import_may_leave_large_weights_remote(
    tmp_path: Path,
) -> None:
    """Small verified evidence remains importable when every missing manifest file is a model."""
    root = tmp_path / "improvement"
    root.mkdir()
    config_path = tmp_path / "improvement.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "expected_train_normalized_sha256": "train-hash",
                "expected_test_normalized_sha256": "test-hash",
            }
        ),
        encoding="utf-8",
    )
    result = {
        "data": {
            "train_normalized_sha256": "train-hash",
            "test_normalized_sha256": "test-hash",
            "train_rows": 2,
            "test_rows": 1,
        },
        "promoted_candidates": ["candidate-a", "candidate-b"],
        "final_test": {"metrics": {}},
    }
    (root / "comparison.json").write_text(json.dumps(result), encoding="utf-8")
    pd.DataFrame(
        {
            "candidate_id": ["candidate-a", "candidate-a", "candidate-b", "candidate-b"],
            "row_index": [0, 1, 0, 1],
        }
    ).to_csv(root / "oof_predictions.csv", index=False)
    pd.DataFrame({"row_index": [0]}).to_csv(root / "test_predictions.csv", index=False)
    lines = []
    for filename in ("comparison.json", "oof_predictions.csv", "test_predictions.csv"):
        payload = (root / filename).read_bytes()
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {filename}")
    lines.append(f"{'0' * 64}  models/candidate-a/seed-42/model.safetensors")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_improvement_bundle(root, config_path)
    assert (
        verify_improvement_bundle(root, config_path, require_models=False)["data"]["test_rows"] == 1
    )
    (root / "test_predictions.csv").write_text("damaged", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_improvement_bundle(root, config_path, require_models=False)


def test_verify_robbert_bundle_rejects_damage_and_accepts_frozen_contract(tmp_path: Path) -> None:
    """Downloaded Colab evidence is accepted only with intact files and frozen split hashes."""
    root = tmp_path / "bundle"
    root.mkdir()
    config = {
        "expected_train_normalized_sha256": "train-hash",
        "expected_test_normalized_sha256": "test-hash",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = {
        "data": {
            "train_normalized_sha256": "train-hash",
            "test_normalized_sha256": "test-hash",
            "test_rows": 2,
        },
        "candidates": [
            {"objective": "multiclass_logistic"},
            {"objective": "coral_ordinal"},
        ],
    }
    (root / "comparison.json").write_text(json.dumps(result), encoding="utf-8")
    pd.DataFrame({"row_index": [0, 1]}).to_csv(root / "test_predictions.csv", index=False)
    lines = []
    for filename in ("comparison.json", "test_predictions.csv"):
        payload = (root / filename).read_bytes()
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {filename}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert verify_bundle(root, config_path)["data"]["test_rows"] == 2
    (root / "checksums.sha256").write_text(
        "\n".join(lines) + "\n" + f"{'0' * 64}  models/multiclass_logistic/model.safetensors\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_bundle(root, config_path)
    assert verify_bundle(root, config_path, require_models=False)["data"]["test_rows"] == 2
    (root / "test_predictions.csv").write_text("damaged", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_bundle(root, config_path)
