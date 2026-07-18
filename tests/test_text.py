import pytest

from dutch_sentiment.text import normalize_text


def test_normalization_handles_transport_artifacts_and_is_idempotent() -> None:
    raw = "  Dit\u200b is niet slecht.<br />Echt&nbsp;goed!  "
    normalized = normalize_text(raw)
    assert normalized == "Dit is niet slecht. Echt goed!"
    assert normalize_text(normalized) == normalized
    assert "niet" in normalized


@pytest.mark.parametrize(
    "review",
    ["Ik geef deze film 8/10.", "Mijn oordeel: 3 out of 10.", "Score: ****/*****"],
)
def test_rating_masking(review: str) -> None:
    masked = normalize_text(review, mask_ratings=True)
    assert "RATINGTOKEN" in masked
    assert normalize_text(masked, mask_ratings=True) == masked


def test_normalizer_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        normalize_text(123)  # type: ignore[arg-type]
