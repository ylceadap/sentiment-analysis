from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dutch_sentiment.api import MAX_REVIEW_CHARACTERS, create_app
from dutch_sentiment.constants import ENGLISH_RELIABILITY_WARNING
from dutch_sentiment.service import NonDutchReviewError, PredictionResult


class FakeService:
    model_version = "fake-v1"

    def classify(self, review: str, *, explain: bool = False) -> PredictionResult:
        if "French" in review:
            raise NonDutchReviewError("The review was confidently detected as unsupported.")
        if "explode" in review:
            raise RuntimeError("sensitive internal message")
        is_english = "English" in review
        result = PredictionResult(
            label="Positive",
            model_version=self.model_version,
            detected_language="english" if is_english else "dutch",
            probabilities={"Positive": 0.8, "Average": 0.15, "Negative": 0.05},
            latency_ms=1.25,
            warnings=(ENGLISH_RELIABILITY_WARNING,) if is_english else (),
        )
        if explain:
            return replace(result, explanation={"supporting_word_features": []})
        return result


@pytest.fixture()
def client() -> TestClient:
    with TestClient(create_app(service=FakeService())) as test_client:
        yield test_client


def test_health_and_successful_classification(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_version": "fake-v1", "model_ready": True}
    response = client.post(
        "/classify", json={"review": "Deze film was verrassend goed.", "explain": True}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "Positive"
    assert body["detected_language"] == "dutch"
    assert body["warnings"] == []
    assert body["explanation"] == {"supporting_word_features": []}


def test_root_serves_interactive_web_app(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Dutch Movie Review Sentiment" in response.text
    assert "/static/app.js" in response.text


def test_openapi_documents_classification_contract_and_examples(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    operation = schema["paths"]["/classify"]["post"]
    request_schema = schema["components"]["schemas"]["ClassifyRequest"]
    response_schema = schema["components"]["schemas"]["ClassifyResponse"]

    assert operation["summary"] == "Classify one movie review"
    assert request_schema["examples"][0]["explain"] is False
    assert "review" in request_schema["examples"][0]
    assert response_schema["examples"][0]["label"] in {"Positive", "Average", "Negative"}
    assert set(response_schema["examples"][0]["probabilities"]) == {
        "Positive",
        "Average",
        "Negative",
    }


@pytest.mark.parametrize("review", ["", "   \n\t"])
def test_empty_input_rejected(client: TestClient, review: str) -> None:
    assert client.post("/classify", json={"review": review}).status_code == 422


def test_missing_and_oversized_input_rejected(client: TestClient) -> None:
    assert client.post("/classify", json={}).status_code == 422
    response = client.post("/classify", json={"review": "a" * (MAX_REVIEW_CHARACTERS + 1)})
    assert response.status_code == 422


def test_english_is_accepted_with_warning_and_other_errors_are_safe(
    client: TestClient,
) -> None:
    english = client.post("/classify", json={"review": "English review text"})
    assert english.status_code == 200
    assert english.json()["detected_language"] == "english"
    assert english.json()["warnings"] == [ENGLISH_RELIABILITY_WARNING]
    unsupported = client.post("/classify", json={"review": "French review text"})
    assert unsupported.status_code == 422
    internal = client.post("/classify", json={"review": "laat explode gebeuren"})
    assert internal.status_code == 500
    assert "sensitive" not in internal.text


def test_missing_model_fails_during_startup(tmp_path: Path) -> None:
    app = create_app(model_path=tmp_path / "missing.joblib")
    with (
        pytest.raises(FileNotFoundError, match="Model artifact not found"),
        TestClient(app),
    ):
        pass
