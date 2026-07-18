from pathlib import Path

import pytest

from dutch_sentiment.predict import predict_review


def test_predict_review_validates_before_loading_model(tmp_path: Path) -> None:
    missing = tmp_path / "missing.joblib"
    with pytest.raises(ValueError, match="empty"):
        predict_review("  ", model_path=missing)
    with pytest.raises(ValueError, match="8000-character"):
        predict_review("a" * 8001, model_path=missing)
