"""Shared typing contracts for models accepted by the inference service."""

from __future__ import annotations

from typing import Protocol

from .classical import ModelInference


class ClassifierModel(Protocol):
    """Describe the minimal immutable classifier contract used at serving time."""

    version: str

    def infer(self, review: str, *, explain: bool = False) -> ModelInference:
        """Return one label, probability mapping, and optional explanation."""
        ...
