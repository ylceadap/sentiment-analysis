from pathlib import Path

import pandas as pd

from dutch_sentiment.audit import audit_dataset, render_markdown
from dutch_sentiment.config import load_config
from dutch_sentiment.constants import ENGLISH_RELIABILITY_WARNING
from dutch_sentiment.language import LanguageResult, LanguageStatus
from dutch_sentiment.metrics import classification_metrics
from dutch_sentiment.model import ModelInference
from dutch_sentiment.service import InferenceService, NonDutchReviewError


class FixedDetector:
    minimum_dutch_confidence = 0.7
    minimum_margin = 0.2
    short_text_characters = 20

    def detect(self, text: str) -> LanguageResult:
        if text.startswith("ShortEnglish"):
            return LanguageResult(LanguageStatus.AMBIGUOUS, "english", 0.2, 0.6, 0.1)
        if text.startswith("English"):
            return LanguageResult(LanguageStatus.NON_DUTCH, "english", 0.01, 0.99, 0.98)
        if text.startswith("French"):
            return LanguageResult(LanguageStatus.NON_DUTCH, "french", 0.01, 0.99, 0.98)
        return LanguageResult(LanguageStatus.DUTCH, "dutch", 0.99, 0.99, 0.98)


class FixedModel:
    version = "fixed-v1"

    def predict(self, reviews: list[str]) -> list[str]:
        return ["Positive" for _ in reviews]

    def predict_proba(self, reviews: list[str]) -> list[dict[str, float]]:
        return [{"Positive": 0.8, "Average": 0.15, "Negative": 0.05} for _ in reviews]

    def explain(self, review: str) -> dict[str, str]:
        return {"feature": review.split()[0]}

    def infer(self, review: str, *, explain: bool = False) -> ModelInference:
        explanation = self.explain(review) if explain else None
        return ModelInference(
            "Positive",
            {"Positive": 0.8, "Average": 0.15, "Negative": 0.05},
            explanation,
        )


def test_audit_profiles_complete_fixture_and_renders_report(tmp_path: Path) -> None:
    path = tmp_path / "reviews.csv"
    pd.DataFrame(
        {
            "Reviews": [
                "Prachtige film<br />8/10",
                "Prachtige film<br />8/10",
                "Gewoon gemiddeld\u200b verhaal",
                "English movie review with enough words to detect",
                "Vreselijk en saai einde",
                "Slecht acteerwerk en een matig verhaal",
            ],
            "Label": ["Positive", "Positive", "Average", "Average", "Negative", "Negative"],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")
    evidence = audit_dataset(path, FixedDetector())  # type: ignore[arg-type]
    assert evidence["schema"]["rows"] == 6
    assert evidence["duplicates"]["exact_extra_rows"] == 1
    assert evidence["artifacts"]["html_break"] == 2
    assert evidence["language"]["status_counts"] == {"dutch": 5, "non_dutch": 1}
    markdown = render_markdown(evidence)
    assert "# Data Audit" in markdown
    assert "sequential split" in markdown


def test_metrics_include_minority_class_and_confusion_matrix() -> None:
    actual = ["Positive", "Average", "Negative", "Negative"]
    predicted = ["Positive", "Positive", "Negative", "Average"]
    probabilities = [
        {"Positive": 0.8, "Average": 0.1, "Negative": 0.1},
        {"Positive": 0.6, "Average": 0.3, "Negative": 0.1},
        {"Positive": 0.1, "Average": 0.1, "Negative": 0.8},
        {"Positive": 0.1, "Average": 0.6, "Negative": 0.3},
    ]
    metrics = classification_metrics(actual, predicted, probabilities)
    assert metrics["per_class"]["Negative"]["support"] == 2.0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 1, 1]]
    assert metrics["log_loss"] > 0
    assert 0 <= metrics["expected_calibration_error_10_bin"] <= 1
    assert 0 <= metrics["mean_prediction_confidence"] <= 1


def test_inference_service_accepts_english_with_warning_and_rejects_unsupported() -> None:
    service = InferenceService(FixedModel(), FixedDetector())  # type: ignore[arg-type]
    result = service.classify("Deze film is heel goed", explain=True)
    assert result.label == "Positive"
    assert result.model_version == "fixed-v1"
    assert result.detected_language == "dutch"
    assert result.warnings == ()
    assert result.explanation == {"feature": "Deze"}
    english = service.classify("English review with obvious language")
    assert english.detected_language == "english"
    assert english.warnings == (ENGLISH_RELIABILITY_WARNING,)
    short_english = service.classify("ShortEnglish")
    assert short_english.detected_language == "english"
    assert short_english.warnings == (ENGLISH_RELIABILITY_WARNING,)
    try:
        service.classify("French review with obvious language")
    except NonDutchReviewError as exc:
        assert "unsupported language" in str(exc)
    else:
        raise AssertionError("Expected unsupported-language input to be rejected")


def test_load_config_accepts_mapping_and_rejects_non_mapping(tmp_path: Path) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text("seed: 42\n", encoding="utf-8")
    assert load_config(valid) == {"seed": 42}
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- one\n- two\n", encoding="utf-8")
    try:
        load_config(invalid)
    except ValueError as exc:
        assert "YAML mapping" in str(exc)
    else:
        raise AssertionError("Expected non-mapping configuration to fail")
