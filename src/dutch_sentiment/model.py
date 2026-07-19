"""Compatibility imports for the classical model implementation."""

from .models.classical import ModelInference, ModelSpec, SentimentModel, build_pipeline
from .models.ordinal_classical import (
    OrdinalSentimentModel,
    SupportedSentimentModel,
    build_ordinal_model,
    load_sentiment_model,
)

__all__ = [
    "ModelInference",
    "ModelSpec",
    "OrdinalSentimentModel",
    "SentimentModel",
    "SupportedSentimentModel",
    "build_ordinal_model",
    "build_pipeline",
    "load_sentiment_model",
]
