"""RobBERT multiclass and rank-consistent ordinal classification components."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import LABELS
from .ordinal import ORDERED_LABELS, compose_ordinal_probabilities

OBJECTIVES = ("multiclass_logistic", "coral_ordinal")


def multiclass_targets(labels: list[str]) -> np.ndarray:
    """Map project labels to the fixed softmax class order."""
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown sentiment labels: {unknown}")
    return np.asarray([LABELS.index(label) for label in labels], dtype=np.int64)


def ordinal_targets(labels: list[str]) -> np.ndarray:
    """Encode each ordered label as the two cumulative targets y>Negative and y>Average."""
    unknown = sorted(set(labels) - set(ORDERED_LABELS))
    if unknown:
        raise ValueError(f"Unknown sentiment labels: {unknown}")
    ranks = np.asarray([ORDERED_LABELS.index(label) for label in labels], dtype=np.int64)
    return np.column_stack((ranks > 0, ranks > 1)).astype(np.float32)


def balanced_class_weights(labels: list[str]) -> np.ndarray:
    """Return inverse-frequency multiclass weights computed only from training labels."""
    targets = multiclass_targets(labels)
    counts = np.bincount(targets, minlength=len(LABELS)).astype(float)
    if np.any(counts == 0):
        raise ValueError("Every sentiment class must occur in the training labels")
    return len(targets) / (len(LABELS) * counts)


def balanced_boundary_weights(labels: list[str]) -> np.ndarray:
    """Return per-boundary negative/positive weights for the two CORAL binary targets."""
    targets = ordinal_targets(labels).astype(np.int64)
    weights = np.empty((2, 2), dtype=np.float32)
    for boundary in range(2):
        counts = np.bincount(targets[:, boundary], minlength=2).astype(float)
        if np.any(counts == 0):
            raise ValueError("Both outcomes must occur at every ordinal boundary")
        weights[boundary] = len(targets) / (2.0 * counts)
    return weights


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert multiclass logits to normalized probabilities in project label order."""
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError("Multiclass logits must have shape (rows, 3)")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def coral_probabilities(logits: np.ndarray) -> np.ndarray:
    """Compose project-ordered class probabilities from two cumulative CORAL logits."""
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("CORAL logits must have shape (rows, 2)")
    boundaries = 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
    ordered = compose_ordinal_probabilities(boundaries[:, 0], boundaries[:, 1])
    indices = [ORDERED_LABELS.index(label) for label in LABELS]
    return ordered[:, indices]


def labels_from_probabilities(probabilities: np.ndarray) -> list[str]:
    """Choose the maximum-probability label using the shared project order."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError("Probability matrix must have shape (rows, 3)")
    return [LABELS[int(index)] for index in values.argmax(axis=1)]


def build_robbert_model(
    model_id: str, revision: str, objective: str, *, dropout: float = 0.1
) -> Any:
    """Build a lazily imported RobBERT encoder with a multiclass or CORAL head."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unsupported RobBERT objective: {objective}")
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
    from transformers import AutoModel

    class RobBERTClassifier(nn.Module):
        """Attach a task head to the fully trainable RobBERT encoder."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_id, revision=revision)
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            if objective == "multiclass_logistic":
                self.classifier = nn.Linear(hidden_size, len(LABELS))
            else:
                self.score = nn.Linear(hidden_size, 1, bias=False)
                self.first_cutpoint = nn.Parameter(torch.tensor(-0.5))
                self.cutpoint_gap = nn.Parameter(torch.tensor(0.0))

        def forward(self, **inputs: Any) -> Any:
            """Encode the first token and return task logits."""
            encoded = self.encoder(**inputs).last_hidden_state[:, 0]
            pooled = self.dropout(encoded)
            if objective == "multiclass_logistic":
                return self.classifier(pooled)
            score = self.score(pooled)
            second = self.first_cutpoint + torch_functional.softplus(self.cutpoint_gap)
            cutpoints = torch.stack((self.first_cutpoint, second)).reshape(1, 2)
            # score-cutpoint guarantees P(y>Average) <= P(y>Negative).
            return score - cutpoints

    return RobBERTClassifier()


def save_robbert_artifact(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    model_id: str,
    revision: str,
    objective: str,
    max_length: int,
    input_strategy: str = "first",
    loss: str | None = None,
) -> None:
    """Save portable safetensors weights, tokenizer files, and reconstruction metadata."""
    from safetensors.torch import save_file

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    save_file(state, destination / "model.safetensors")
    tokenizer.save_pretrained(destination)
    metadata = {
        "model_id": model_id,
        "revision": revision,
        "objective": objective,
        "max_length": max_length,
        "input_strategy": input_strategy,
        "loss": loss,
        "label_order": list(LABELS),
        "ordered_labels": list(ORDERED_LABELS),
    }
    (destination / "robbert_artifact.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def probability_rows(probabilities: np.ndarray) -> list[dict[str, float]]:
    """Convert a probability matrix to the mapping format shared by project metrics."""
    values = np.asarray(probabilities, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError("Probability matrix must have shape (rows, 3)")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Every probability row must sum to one")
    return [{label: float(row[index]) for index, label in enumerate(LABELS)} for row in values]


def finite_metric(value: float) -> float | None:
    """Represent non-finite diagnostic values safely in JSON evidence."""
    return float(value) if math.isfinite(value) else None
