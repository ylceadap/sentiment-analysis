"""Compatibility imports for the classical model implementation."""

from .models.classical import ModelInference, ModelSpec, SentimentModel, build_pipeline

__all__ = ["ModelInference", "ModelSpec", "SentimentModel", "build_pipeline"]
