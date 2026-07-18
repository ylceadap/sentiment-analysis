"""Thread-safe inference orchestration around immutable model and language components."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .language import DutchLanguageDetector, LanguageStatus
from .model import SentimentModel


class NonDutchReviewError(ValueError):
    """Raised when the local detector confidently identifies non-Dutch input."""


@dataclass(frozen=True)
class PredictionResult:
    label: str
    model_version: str
    detected_language: str
    probabilities: dict[str, float]
    latency_ms: float
    explanation: dict[str, Any] | None = None


class InferenceService:
    """Apply language policy and one loaded immutable model per request."""

    def __init__(self, model: SentimentModel, detector: DutchLanguageDetector) -> None:
        self.model = model
        self.detector = detector

    @property
    def model_version(self) -> str:
        return self.model.version

    def classify(self, review: str, *, explain: bool = False) -> PredictionResult:
        started = perf_counter()
        language = self.detector.detect(review)
        if language.status is LanguageStatus.NON_DUTCH:
            raise NonDutchReviewError(
                "The review was confidently detected as non-Dutch; submit a Dutch review."
            )
        label = self.model.predict([review])[0]
        probabilities = self.model.predict_proba([review])[0]
        explanation = self.model.explain(review) if explain else None
        latency_ms = (perf_counter() - started) * 1000
        detected = (
            "ambiguous"
            if language.status is LanguageStatus.AMBIGUOUS
            else language.detected_language or language.status.value
        )
        return PredictionResult(
            label=label,
            model_version=self.model.version,
            detected_language=detected,
            probabilities=probabilities,
            latency_ms=round(latency_ms, 3),
            explanation=explanation,
        )
