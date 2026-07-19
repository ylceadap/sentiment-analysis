"""FastAPI application exposing the Dutch-primary bilingual sentiment model."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .constants import LABELS, MAX_REVIEW_CHARACTERS
from .language import DutchLanguageDetector
from .models.classical import SentimentModel
from .models.llm_advisor import LLMRecommendationResult, LLMRecommender
from .service import InferenceService, NonDutchReviewError

LOGGER = logging.getLogger(__name__)
Label = Literal["Positive", "Average", "Negative"]


class ClassifyRequest(BaseModel):
    """Validate the public classification request body."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "review": (
                        "Deze film was verrassend goed, met sterke acteurs en een mooi einde."
                    ),
                    "explain": False,
                }
            ]
        },
    )
    review: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_REVIEW_CHARACTERS,
            description="One Dutch or English movie review to classify.",
        ),
    ]
    explain: bool = Field(
        default=False,
        description="Include local linear feature contributions in the response.",
    )

    @field_validator("review")
    @classmethod
    def reject_blank_review(cls, value: str) -> str:
        """Reject whitespace-only reviews after Pydantic length validation."""
        if not value.strip():
            raise ValueError("review must not be empty or whitespace-only")
        return value


class ClassifyResponse(BaseModel):
    """Define the stable successful classification response contract."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "label": "Average",
                    "model_version": "0.1.0+240b3fc",
                    "detected_language": "dutch",
                    "probabilities": {
                        "Positive": 0.319,
                        "Average": 0.547,
                        "Negative": 0.134,
                    },
                    "latency_ms": 5.32,
                    "warnings": [],
                    "explanation": None,
                }
            ]
        }
    )
    label: Label = Field(description="The predicted sentiment label.")
    model_version: str = Field(description="The version of the loaded model artifact.")
    detected_language: str = Field(description="The language detected for the submitted review.")
    probabilities: dict[Label, float] = Field(
        description="Native Logistic Regression probabilities for all supported labels."
    )
    latency_ms: float = Field(description="End-to-end service inference latency in milliseconds.")
    warnings: tuple[str, ...] = Field(
        default=(), description="Reliability or input-policy warnings."
    )
    explanation: dict[str, Any] | None = Field(
        default=None,
        description="Optional local linear feature contributions when explain is true.",
    )


class HealthResponse(BaseModel):
    """Define readiness evidence returned by the health endpoint."""

    status: Literal["ok"]
    model_version: str
    model_ready: Literal[True]


class LLMRecommendationResponse(BaseModel):
    """Describe the optional external LLM advisory result."""

    status: Literal["ok", "unavailable", "error"]
    provider: str
    model: str
    prompt_profile: str = Field(
        description="Versioned runtime prompt; historical 24-shot evidence uses another profile."
    )
    label: Label | None = None
    rationale: str | None = None
    confidence: float | None = None
    latency_ms: float | None = None
    warning: str | None = None


class RecommendationResponse(BaseModel):
    """Return the formal model prediction beside optional LLM advice."""

    model_prediction: ClassifyResponse
    llm_recommendation: LLMRecommendationResponse
    agreement: bool | None = Field(
        description="Whether both systems returned the same label; null when LLM is unavailable."
    )


def create_app(
    *,
    service: InferenceService | Any | None = None,
    llm_recommender: LLMRecommender | Any | None = None,
    model_path: str | Path | None = None,
) -> FastAPI:
    """Build an app that can inject a lightweight service in API tests."""
    resolved_path = Path(model_path or os.getenv("MODEL_PATH", "artifacts/model.joblib"))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load and warm production components before marking the app ready."""
        if service is None:
            model = SentimentModel.load(resolved_path)
            detector = DutchLanguageDetector()
            warmup_text = "Dit is een Nederlandse tekst om de taalmodellen op te warmen."
            detector.detect(warmup_text)
            model.infer(warmup_text, explain=True)
            app.state.service = InferenceService(model, detector)
        else:
            app.state.service = service
        app.state.llm_recommender = llm_recommender or LLMRecommender.from_environment()
        LOGGER.info("model_loaded version=%s", app.state.service.model_version)
        yield

    app = FastAPI(
        title="Dutch-primary Movie Review Sentiment",
        version=__version__,
        lifespan=lifespan,
    )
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the bundled interactive review interface."""
        return FileResponse(static_dir / "index.html")

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Report readiness and the loaded model version."""
        return HealthResponse(
            status="ok", model_version=request.app.state.service.model_version, model_ready=True
        )

    @app.post(
        "/classify",
        response_model=ClassifyResponse,
        summary="Classify one movie review",
        description=(
            "Apply language detection, the fitted text transformations, and the sentiment model "
            "to one Dutch or English movie review."
        ),
        response_description="The predicted label and supporting inference metadata.",
    )
    async def classify(payload: ClassifyRequest, request: Request) -> ClassifyResponse:
        """Classify one review while returning safe public errors."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        try:
            result = request.app.state.service.classify(payload.review, explain=payload.explain)
        except NonDutchReviewError as exc:
            LOGGER.info(
                "classification_rejected request_id=%s input_length=%d category=unsupported_language",
                request_id,
                len(payload.review),
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception(
                "classification_failed request_id=%s input_length=%d category=internal",
                request_id,
                len(payload.review),
            )
            raise HTTPException(
                status_code=500, detail="Classification failed due to an internal error."
            ) from exc
        if result.label not in LABELS:
            raise HTTPException(status_code=500, detail="Model returned an invalid label.")
        LOGGER.info(
            "classification_complete request_id=%s input_length=%d language=%s label=%s latency_ms=%.3f",
            request_id,
            len(payload.review),
            result.detected_language,
            result.label,
            result.latency_ms,
        )
        return ClassifyResponse(**result.__dict__)

    @app.post(
        "/recommendations",
        response_model=RecommendationResponse,
        summary="Compare model and LLM recommendations",
        description=(
            "Return the reproducible submitted model prediction and, when configured, an "
            "advisory LLM sentiment recommendation for the same review."
        ),
    )
    async def recommendations(payload: ClassifyRequest, request: Request) -> RecommendationResponse:
        """Compare the formal classifier with the optional advisory LLM."""
        model_prediction = await classify(payload, request)
        llm_result: LLMRecommendationResult = request.app.state.llm_recommender.recommend(
            payload.review,
            detected_language=model_prediction.detected_language,
        )
        agreement = (
            model_prediction.label == llm_result.label if llm_result.status == "ok" else None
        )
        return RecommendationResponse(
            model_prediction=model_prediction,
            llm_recommendation=LLMRecommendationResponse(**llm_result.__dict__),
            agreement=agreement,
        )

    return app


def run() -> None:
    """Start the Uvicorn server using environment-configured port settings."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
