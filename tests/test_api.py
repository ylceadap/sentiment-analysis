from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from dutch_sentiment.api import MAX_REVIEW_CHARACTERS, create_app
from dutch_sentiment.constants import ENGLISH_RELIABILITY_WARNING
from dutch_sentiment.llm_recommender import LLMRecommendationResult
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


class FakeLLMRecommender:
    def recommend(self, review: str, *, detected_language: str) -> LLMRecommendationResult:
        return LLMRecommendationResult(
            status="ok",
            provider="fake",
            model="fake-llm",
            prompt_profile="fake-zero-shot-v1",
            label="Positive",
            rationale=f"Matched {detected_language} positive language.",
            confidence=0.77,
            latency_ms=2.5,
            warning="LLM output is advisory only.",
        )


@pytest.fixture()
def anyio_backend() -> str:
    """Run async API tests on the installed asyncio backend only."""
    return "asyncio"


@pytest.fixture()
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Yield an in-process HTTP client with application lifespan enabled."""
    app = create_app(service=FakeService(), llm_recommender=FakeLLMRecommender())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
            yield test_client


@pytest.mark.anyio
async def test_health_and_successful_classification(client: httpx.AsyncClient) -> None:
    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_version": "fake-v1", "model_ready": True}
    response = await client.post(
        "/classify", json={"review": "Deze film was verrassend goed.", "explain": True}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "Positive"
    assert body["detected_language"] == "dutch"
    assert body["warnings"] == []
    assert body["explanation"] == {"supporting_word_features": []}


@pytest.mark.anyio
async def test_root_serves_interactive_web_app(client: httpx.AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    assert "Model and LLM Review" in response.text
    assert "Five-model held-out comparison" in response.text
    assert "Live UI: Production TF-IDF" in response.text
    assert "/static/app.js" in response.text


@pytest.mark.anyio
async def test_model_comparison_exposes_evidence_not_live_model_selection(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/model-comparison")
    assert response.status_code == 200
    body = response.json()
    assert body["evaluation_scope"] == "reused-heldout-presentation-comparison"
    assert body["heldout_rows"] == 960
    assert len(body["ranking"]) == 5
    assert body["production_model"] == "Current Production TF-IDF"
    assert body["ranking"][0]["model"] == "DeepSeek V4 Flash 24-shot"


@pytest.mark.anyio
async def test_recommendations_returns_model_and_llm_advice(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/recommendations", json={"review": "Deze film was verrassend goed.", "explain": False}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_prediction"]["label"] == "Positive"
    assert body["llm_recommendation"]["status"] == "ok"
    assert body["llm_recommendation"]["label"] == "Positive"
    assert body["llm_recommendation"]["prompt_profile"] == "fake-zero-shot-v1"
    assert body["agreement"] is True


@pytest.mark.anyio
async def test_openapi_documents_classification_contract_and_examples(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/openapi.json")
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
@pytest.mark.anyio
async def test_empty_input_rejected(client: httpx.AsyncClient, review: str) -> None:
    assert (await client.post("/classify", json={"review": review})).status_code == 422


@pytest.mark.anyio
async def test_missing_and_oversized_input_rejected(client: httpx.AsyncClient) -> None:
    assert (await client.post("/classify", json={})).status_code == 422
    response = await client.post("/classify", json={"review": "a" * (MAX_REVIEW_CHARACTERS + 1)})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_english_is_accepted_with_warning_and_other_errors_are_safe(
    client: httpx.AsyncClient,
) -> None:
    english = await client.post("/classify", json={"review": "English review text"})
    assert english.status_code == 200
    assert english.json()["detected_language"] == "english"
    assert english.json()["warnings"] == [ENGLISH_RELIABILITY_WARNING]
    unsupported = await client.post("/classify", json={"review": "French review text"})
    assert unsupported.status_code == 422
    internal = await client.post("/classify", json={"review": "laat explode gebeuren"})
    assert internal.status_code == 500
    assert "sensitive" not in internal.text


@pytest.mark.anyio
async def test_missing_model_fails_during_startup(tmp_path: Path) -> None:
    app = create_app(model_path=tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        async with app.router.lifespan_context(app):
            pass
