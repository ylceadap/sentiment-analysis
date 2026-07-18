import pytest

from dutch_sentiment.language import DutchLanguageDetector, LanguageStatus


@pytest.fixture(scope="module")
def detector() -> DutchLanguageDetector:
    return DutchLanguageDetector()


def test_clear_dutch_and_english(detector: DutchLanguageDetector) -> None:
    dutch = detector.detect(
        "Deze film heeft een prachtig verhaal, sterke acteurs en een verrassend goed einde."
    )
    english = detector.detect(
        "This movie has a predictable story, weak acting, and an extremely disappointing ending."
    )
    assert dutch.status is LanguageStatus.DUTCH
    assert dutch.detected_language == "dutch"
    assert english.status is LanguageStatus.NON_DUTCH
    assert english.detected_language == "english"


def test_short_text_is_ambiguous_and_deterministic(detector: DutchLanguageDetector) -> None:
    first = detector.detect("Goed")
    second = detector.detect("Goed")
    assert first.status is LanguageStatus.AMBIGUOUS
    assert first == second
