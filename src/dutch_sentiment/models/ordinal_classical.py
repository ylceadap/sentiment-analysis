"""Serializable TF-IDF ordinal model retained for archived artifact compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..constants import LABELS
from .classical import ModelInference, ModelSpec, SentimentModel, build_pipeline


class OrdinalSentimentModel:
    """Serve two calibrated cumulative boundaries through the common model contract."""

    def __init__(
        self,
        feature_pipeline: Pipeline,
        lower_classifier: CalibratedClassifierCV,
        upper_classifier: CalibratedClassifierCV,
        *,
        lower_threshold: float,
        upper_threshold: float,
        version: str = "unversioned",
    ) -> None:
        """Store the shared features, boundary classifiers, and frozen decision thresholds."""
        self.feature_pipeline = feature_pipeline
        self.lower_classifier = lower_classifier
        self.upper_classifier = upper_classifier
        self.lower_threshold = float(lower_threshold)
        self.upper_threshold = float(upper_threshold)
        self.version = version
        self._feature_names_cache: Any | None = None

    def fit(self, reviews: list[str], labels: list[str]) -> OrdinalSentimentModel:
        """Fit shared TF-IDF features and both cumulative binary boundaries."""
        unexpected = sorted(set(labels) - set(LABELS))
        if unexpected:
            raise ValueError(f"Unexpected labels: {unexpected}")
        transformed = self.feature_pipeline.fit_transform(reviews)
        values = np.asarray(labels)
        self.lower_classifier.fit(transformed, (values != "Negative").astype(int))
        self.upper_classifier.fit(transformed, (values == "Positive").astype(int))
        return self

    @staticmethod
    def _boundary_probabilities(
        lower_classifier: CalibratedClassifierCV,
        upper_classifier: CalibratedClassifierCV,
        transformed: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return monotonic P(y>Negative) and P(y>Average) boundaries."""
        lower = lower_classifier.predict_proba(transformed)[:, 1]
        upper = upper_classifier.predict_proba(transformed)[:, 1]
        violations = upper > lower
        # The midpoint is the least-squares projection when upper exceeds lower.
        midpoint = (lower[violations] + upper[violations]) / 2.0
        lower[violations] = midpoint
        upper[violations] = midpoint
        return lower, upper

    @staticmethod
    def _probability_rows(lower: np.ndarray, upper: np.ndarray) -> list[dict[str, float]]:
        """Compose P(N)=1-lower, P(A)=lower-upper, and P(P)=upper."""
        return [
            {"Positive": float(p), "Average": float(a), "Negative": float(n)}
            for n, a, p in zip(1.0 - lower, lower - upper, upper, strict=True)
        ]

    def _labels(self, lower: np.ndarray, upper: np.ndarray) -> list[str]:
        """Apply the frozen lower and upper deployment thresholds."""
        return np.where(
            lower < self.lower_threshold,
            "Negative",
            np.where(upper >= self.upper_threshold, "Positive", "Average"),
        ).tolist()

    def predict(self, reviews: list[str]) -> list[str]:
        """Predict ordered sentiment labels for a batch of reviews."""
        transformed = self.feature_pipeline.transform(reviews)
        lower, upper = self._boundary_probabilities(
            self.lower_classifier, self.upper_classifier, transformed
        )
        return self._labels(lower, upper)

    def predict_proba(self, reviews: list[str]) -> list[dict[str, float]]:
        """Return composed class probabilities for a batch of reviews."""
        transformed = self.feature_pipeline.transform(reviews)
        lower, upper = self._boundary_probabilities(
            self.lower_classifier, self.upper_classifier, transformed
        )
        return self._probability_rows(lower, upper)

    def infer(self, review: str, *, explain: bool = False, top_n: int = 5) -> ModelInference:
        """Return one label and probability row; archived explanations are unsupported."""
        if explain:
            raise RuntimeError("Ordinal archived-artifact explanations are not supported")
        probabilities = self.predict_proba([review])[0]
        label = self.predict([review])[0]
        return ModelInference(label, probabilities, None)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted ordinal model with compression."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output, compress=3)


SupportedSentimentModel = SentimentModel | OrdinalSentimentModel


def build_ordinal_model(
    config: dict[str, Any],
    *,
    c_value: float,
    calibration_folds: int,
    lower_threshold: float,
    upper_threshold: float,
    version: str,
) -> OrdinalSentimentModel:
    """Build an unfitted ordinal model using the submitted sparse text features."""
    base = build_pipeline(ModelSpec("ordinal_logistic", "combined", "balanced"), config)
    features = Pipeline(base.steps[:-1])

    def classifier() -> CalibratedClassifierCV:
        """Build one balanced sigmoid-calibrated boundary classifier."""
        estimator = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=int(config.get("max_iter", 1500)),
            random_state=int(config.get("random_seed", 42)),
            solver="lbfgs",
        )
        return CalibratedClassifierCV(estimator, method="sigmoid", cv=calibration_folds, n_jobs=1)

    return OrdinalSentimentModel(
        features,
        classifier(),
        classifier(),
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        version=version,
    )


def load_sentiment_model(path: str | Path) -> SupportedSentimentModel:
    """Load either supported model type from a trusted local artifact."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    loaded = joblib.load(model_path)
    if not isinstance(loaded, (SentimentModel, OrdinalSentimentModel)):
        raise TypeError(f"Artifact is not a supported sentiment model: {model_path}")
    return loaded
