"""Repeatable cold/warm component, service, explanation, and HTTP latency benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from .api import create_app
from .data import sha256_file
from .language import DutchLanguageDetector
from .model import load_sentiment_model
from .reporting import build_model_report
from .service import InferenceService

SAMPLES = [
    "Deze film was verrassend goed, met overtuigende acteurs en een sterk einde.",
    (
        "Hoewel het verhaal soms voorspelbaar was, bleven de personages geloofwaardig en "
        "zorgden de sterke dialogen en mooie beelden voor een boeiende Nederlandse filmervaring."
    ),
    " ".join(
        [
            "De film begint rustig en bouwt de spanning zorgvuldig op, maar verliest halverwege "
            "tempo voordat het ontroerende slot de belangrijkste verhaallijnen overtuigend afrondt."
        ]
        * 10
    ),
]


def _measure(function: Callable[[str], object], iterations: int, warmup: int) -> dict[str, Any]:
    for index in range(warmup):
        function(SAMPLES[index % len(SAMPLES)])
    values = []
    for index in range(iterations):
        started = perf_counter()
        function(SAMPLES[index % len(SAMPLES)])
        values.append((perf_counter() - started) * 1000)
    return {
        "iterations": iterations,
        "mean_ms": statistics.fmean(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": max(values),
    }


def run_benchmark(
    model_path: str | Path, *, iterations: int = 100, warmup: int = 10
) -> dict[str, Any]:
    model_path = Path(model_path)
    started = perf_counter()
    detector = DutchLanguageDetector()
    detector_initialization_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    detector.detect(SAMPLES[0])
    cold_detector_first_inference_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    model = load_sentiment_model(model_path)
    cold_model_load_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    model.predict([SAMPLES[0]])
    cold_first_prediction_ms = (perf_counter() - started) * 1000
    service = InferenceService(model, detector)

    measurements = {
        "normalization_plus_model_prediction": _measure(
            lambda text: model.predict([text]), iterations, warmup
        ),
        "language_detection_plus_model": _measure(
            lambda text: (detector.detect(text), model.predict([text])), iterations, warmup
        ),
        "service_end_to_end": _measure(
            lambda text: service.classify(text, explain=False), iterations, warmup
        ),
        "explanation_enabled": _measure(
            lambda text: service.classify(text, explain=True), iterations, warmup
        ),
    }
    app = create_app(service=service)
    with TestClient(app) as client:
        measurements["http_classify"] = _measure(
            lambda text: client.post("/classify", json={"review": text}).raise_for_status(),
            max(30, iterations // 2),
            min(5, warmup),
        )
    return {
        "environment": {
            "operating_system": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor() or "not reported",
        },
        "configuration": {"iterations": iterations, "warmup": warmup, "sample_count": len(SAMPLES)},
        "detector_initialization_ms": detector_initialization_ms,
        "cold_detector_first_inference_ms": cold_detector_first_inference_ms,
        "cold_model_load_ms": cold_model_load_ms,
        "cold_first_prediction_ms": cold_first_prediction_ms,
        "model_artifact_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--output", default="artifacts/benchmark.json")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    result = run_benchmark(args.model, iterations=args.iterations, warmup=args.warmup)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    build_model_report(output.parent, "reports/model_report.md")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
