"""Thread-safe inference orchestration around immutable model and language components."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .constants import ENGLISH_RELIABILITY_WARNING
from .language import DutchLanguageDetector, LanguageStatus
from .model import SentimentModel


class NonDutchReviewError(ValueError):
    """Raised when the detector confidently identifies an unsupported language."""


@dataclass(frozen=True)
class PredictionResult:
    """Contain the complete stable response returned by the inference service."""

    label: str
    model_version: str
    detected_language: str
    probabilities: dict[str, float]
    latency_ms: float
    warnings: tuple[str, ...] = ()
    explanation: dict[str, Any] | None = None


class InferenceService:
    """Apply language policy and one loaded immutable model per request."""

    def __init__(self, model: SentimentModel, detector: DutchLanguageDetector) -> None:
        """Bind one immutable fitted model to one local language detector."""
        self.model = model
        self.detector = detector

    @property
    def model_version(self) -> str:
        """Expose the version embedded in the loaded model artifact."""
        return self.model.version

    def classify(self, review: str, *, explain: bool = False) -> PredictionResult:
        """Apply language policy and return one timed model inference."""
        started = perf_counter()
        language = self.detector.detect(review)
        is_english = language.detected_language == "english"
        if language.status is LanguageStatus.NON_DUTCH and not is_english:
            raise NonDutchReviewError(
                "The review was confidently detected as an unsupported language; "
                "submit a Dutch or English review."
            )
        inference = self.model.infer(review, explain=explain)
        latency_ms = (perf_counter() - started) * 1000
        detected = language.detected_language or "ambiguous"
        warnings = (ENGLISH_RELIABILITY_WARNING,) if is_english else ()
        return PredictionResult(
            label=inference.label,
            model_version=self.model.version,
            detected_language=detected,
            probabilities=inference.probabilities,
            latency_ms=round(latency_ms, 3),
            warnings=warnings,
            explanation=inference.explanation,
        )
