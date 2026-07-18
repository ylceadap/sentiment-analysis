"""Deterministic local language identification with an explicit ambiguity policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

from .text import normalize_text


class LanguageStatus(StrEnum):
    """Represent accepted Dutch, unsupported, or ambiguous language evidence."""

    DUTCH = "dutch"
    NON_DUTCH = "non_dutch"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class LanguageResult:
    """Contain local detector scores and the resulting policy status."""

    status: LanguageStatus
    detected_language: str | None
    dutch_confidence: float
    top_confidence: float
    margin: float


class DutchLanguageDetector:
    """Classify review language locally without using sentiment labels or hosted APIs."""

    _languages = (
        Language.DUTCH,
        Language.ENGLISH,
        Language.GERMAN,
        Language.FRENCH,
        Language.SPANISH,
        Language.ITALIAN,
        Language.PORTUGUESE,
    )

    def __init__(
        self,
        minimum_dutch_confidence: float = 0.70,
        minimum_margin: float = 0.20,
        short_text_characters: int = 20,
    ) -> None:
        """Configure confidence, margin, and short-text ambiguity thresholds."""
        self.minimum_dutch_confidence = minimum_dutch_confidence
        self.minimum_margin = minimum_margin
        self.short_text_characters = short_text_characters
        self._detector: LanguageDetector = LanguageDetectorBuilder.from_languages(
            *self._languages
        ).build()

    def detect(self, text: str) -> LanguageResult:
        """Detect review language and apply the explicit ambiguity policy."""
        cleaned = normalize_text(text)
        values = self._detector.compute_language_confidence_values(cleaned)
        if not values:
            return LanguageResult(LanguageStatus.AMBIGUOUS, None, 0.0, 0.0, 0.0)
        scores = {value.language: float(value.value) for value in values}
        top = values[0]
        second_score = float(values[1].value) if len(values) > 1 else 0.0
        dutch_score = round(scores.get(Language.DUTCH, 0.0), 12)
        top_score = round(float(top.value), 12)
        margin = round(top_score - round(second_score, 12), 12)
        detected = top.language.name.lower()

        if len(cleaned) < self.short_text_characters:
            return LanguageResult(
                LanguageStatus.AMBIGUOUS,
                detected,
                dutch_score,
                top_score,
                margin,
            )
        if top.language == Language.DUTCH:
            status = (
                LanguageStatus.DUTCH
                if dutch_score >= self.minimum_dutch_confidence and margin >= self.minimum_margin
                else LanguageStatus.AMBIGUOUS
            )
        else:
            status = (
                LanguageStatus.NON_DUTCH
                if top_score >= self.minimum_dutch_confidence and margin >= self.minimum_margin
                else LanguageStatus.AMBIGUOUS
            )
        return LanguageResult(status, detected, dutch_score, top_score, margin)
