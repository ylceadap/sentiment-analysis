"""Compatibility aliases required to load previously published model artifacts."""

from .models.classical import SentimentModel
from .models.ordinal_classical import OrdinalSentimentModel

__all__ = ["OrdinalSentimentModel", "SentimentModel"]
