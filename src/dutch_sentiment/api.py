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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import LABELS, MAX_REVIEW_CHARACTERS
from .language import DutchLanguageDetector
from .model import SentimentModel
from .service import InferenceService, NonDutchReviewError

LOGGER = logging.getLogger(__name__)
Label = Literal["Positive", "Average", "Negative"]


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review: Annotated[str, Field(min_length=1, max_length=MAX_REVIEW_CHARACTERS)]
    explain: bool = False

    @field_validator("review")
    @classmethod
    def reject_blank_review(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("review must not be empty or whitespace-only")
        return value


class ClassifyResponse(BaseModel):
    label: Label
    model_version: str
    detected_language: str
    probabilities: dict[Label, float]
    latency_ms: float
    warnings: tuple[str, ...] = ()
    explanation: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    model_ready: Literal[True]


def create_app(
    *,
    service: InferenceService | Any | None = None,
    model_path: str | Path | None = None,
) -> FastAPI:
    """Build an app that can inject a lightweight service in API tests."""
    resolved_path = Path(model_path or os.getenv("MODEL_PATH", "artifacts/model.joblib"))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            model = SentimentModel.load(resolved_path)
            detector = DutchLanguageDetector()
            warmup_text = "Dit is een Nederlandse tekst om de taalmodellen op te warmen."
            detector.detect(warmup_text)
            model.infer(warmup_text, explain=True)
            app.state.service = InferenceService(model, detector)
        else:
            app.state.service = service
        LOGGER.info("model_loaded version=%s", app.state.service.model_version)
        yield

    app = FastAPI(
        title="Dutch-primary Movie Review Sentiment",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok", model_version=request.app.state.service.model_version, model_ready=True
        )

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(payload: ClassifyRequest, request: Request) -> ClassifyResponse:
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

    return app


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
