"""Run one local prediction without starting the HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .constants import MAX_REVIEW_CHARACTERS
from .language import DutchLanguageDetector
from .model import SentimentModel
from .service import InferenceService, NonDutchReviewError


def predict_review(
    review: str,
    *,
    model_path: str | Path = "artifacts/model.joblib",
    explain: bool = False,
) -> dict[str, Any]:
    """Load the locally controlled artifact and return one JSON-compatible result."""
    if not review.strip():
        raise ValueError("review must not be empty or whitespace-only")
    if len(review) > MAX_REVIEW_CHARACTERS:
        raise ValueError(f"review exceeds the {MAX_REVIEW_CHARACTERS}-character limit")
    model = SentimentModel.load(model_path)
    detector = DutchLanguageDetector()
    detector.detect("Dit is een Nederlandse tekst om de taalmodellen op te warmen.")
    service = InferenceService(model, detector)
    result = service.classify(review, explain=explain)
    return result.__dict__


def parse_args() -> argparse.Namespace:
    """Parse local prediction CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, help="One Dutch or English movie review")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--explain", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run one validated local prediction and print JSON."""
    args = parse_args()
    try:
        result = predict_review(args.review, model_path=args.model, explain=args.explain)
    except (ValueError, NonDutchReviewError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
