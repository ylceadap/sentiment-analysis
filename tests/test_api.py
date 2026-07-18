from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dutch_sentiment.api import MAX_REVIEW_CHARACTERS, create_app
from dutch_sentiment.service import NonDutchReviewError, PredictionResult


class FakeService:
    model_version = "fake-v1"

    def classify(self, review: str, *, explain: bool = False) -> PredictionResult:
        if "English" in review:
            raise NonDutchReviewError("The review was confidently detected as non-Dutch.")
        if "explode" in review:
            raise RuntimeError("sensitive internal message")
        result = PredictionResult(
            label="Positive",
            model_version=self.model_version,
            detected_language="dutch",
            probabilities={"Positive": 0.8, "Average": 0.15, "Negative": 0.05},
            latency_ms=1.25,
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
    assert body["explanation"] == {"supporting_word_features": []}


@pytest.mark.parametrize("review", ["", "   \n\t"])
def test_empty_input_rejected(client: TestClient, review: str) -> None:
    assert client.post("/classify", json={"review": review}).status_code == 422


def test_missing_and_oversized_input_rejected(client: TestClient) -> None:
    assert client.post("/classify", json={}).status_code == 422
    response = client.post("/classify", json={"review": "a" * (MAX_REVIEW_CHARACTERS + 1)})
    assert response.status_code == 422


def test_non_dutch_and_internal_errors_are_safe(client: TestClient) -> None:
    non_dutch = client.post("/classify", json={"review": "English review text"})
    assert non_dutch.status_code == 422
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
