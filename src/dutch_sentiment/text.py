"""Conservative, deterministic review normalization shared by all code paths."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
RATING_RE = re.compile(
    r"(?ix)"
    r"(?:\b(?:10|[0-9](?:[.,][0-9])?)\s*(?:/\s*10|out\s+of\s+10|van(?:\s+de)?\s+10)\b)"
    r"|(?:\*\s*){1,5}/(?:\*\s*){5}"
)


def normalize_text(text: str, *, mask_ratings: bool = False) -> str:
    """Normalize transport artifacts while preserving sentiment-bearing language."""
    if not isinstance(text, str):
        raise TypeError("Review text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = HTML_BREAK_RE.sub("\n", normalized)
    normalized = html.unescape(normalized)
    if mask_ratings:
        normalized = RATING_RE.sub(" RATINGTOKEN ", normalized)
    return WHITESPACE_RE.sub(" ", normalized).strip()


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible normalizer stored inside the fitted model pipeline."""

    def __init__(self, mask_ratings: bool = False) -> None:
        """Store whether explicit rating patterns should be masked."""
        self.mask_ratings = mask_ratings

    def fit(self, x: Iterable[str], y: object = None) -> TextNormalizer:
        """Return the stateless transformer unchanged for sklearn compatibility."""
        del x, y
        return self

    def transform(self, x: Iterable[str]) -> np.ndarray:
        """Normalize every input string into an object array for vectorizers."""
        return np.asarray(
            [normalize_text(value, mask_ratings=self.mask_ratings) for value in x],
            dtype=object,
        )
