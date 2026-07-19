"""Model construction, serialization, prediction, and local linear explanations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ..constants import LABELS
from ..text import TextNormalizer


@dataclass(frozen=True)
class ModelSpec:
    """Describe one transparent classical-model experiment candidate."""

    name: str
    feature_kind: str
    class_weight: str | None = None
    mask_ratings: bool = False
    dummy: bool = False


@dataclass(frozen=True)
class ModelInference:
    """Contain label, probabilities, and optional linear contributions."""

    label: str
    probabilities: dict[str, float]
    explanation: dict[str, Any] | None = None


def build_pipeline(spec: ModelSpec, config: dict[str, Any]) -> Pipeline:
    """Build an unfitted sparse-text pipeline from a transparent experiment spec."""
    normalizer = TextNormalizer(mask_ratings=spec.mask_ratings)
    min_df = int(config.get("min_df", 2))
    max_df = float(config.get("max_df", 0.98))
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=int(config.get("word_max_features", 40000)),
        sublinear_tf=True,
        strip_accents=None,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        max_df=max_df,
        max_features=int(config.get("char_max_features", 50000)),
        sublinear_tf=True,
    )
    if spec.dummy:
        features: Any = TfidfVectorizer(max_features=100, min_df=1)
        classifier: Any = DummyClassifier(strategy="prior")
    else:
        if spec.feature_kind == "word":
            features = word
        elif spec.feature_kind == "char":
            features = char
        elif spec.feature_kind == "combined":
            features = FeatureUnion([("word", word), ("char", char)])
        else:
            raise ValueError(f"Unsupported feature kind: {spec.feature_kind}")
        classifier = LogisticRegression(
            class_weight=spec.class_weight,
            max_iter=int(config.get("max_iter", 1500)),
            random_state=int(config.get("random_seed", 42)),
            solver="lbfgs",
        )
    return Pipeline([("normalize", normalizer), ("features", features), ("classifier", classifier)])


class SentimentModel:
    """Stable application-facing abstraction around a fitted sklearn pipeline."""

    def __init__(self, pipeline: Pipeline, version: str = "unversioned") -> None:
        """Wrap a fitted or unfitted sklearn pipeline with a stable version."""
        self.pipeline = pipeline
        self.version = version
        self._feature_names_cache: Any | None = None

    def fit(self, reviews: list[str], labels: list[str]) -> SentimentModel:
        """Fit the complete normalization, feature, and classifier pipeline."""
        self.pipeline.fit(reviews, labels)
        self._feature_names_cache = None
        return self

    @staticmethod
    def _validate_label(label: str) -> str:
        """Reject labels outside the public three-class contract."""
        if label not in LABELS:
            raise RuntimeError(f"Model returned invalid label: {label}")
        return label

    def predict(self, reviews: list[str]) -> list[str]:
        """Predict and validate labels for a batch of reviews."""
        predictions = self.pipeline.predict(reviews).tolist()
        invalid = sorted(set(predictions) - set(LABELS))
        if invalid:
            raise RuntimeError(f"Model returned invalid labels: {invalid}")
        return predictions

    def _probability_dict(self, transformed: Any) -> dict[str, float]:
        """Map one transformed row's native probabilities to class names."""
        classifier = self.pipeline.named_steps["classifier"]
        if not hasattr(classifier, "predict_proba"):
            raise RuntimeError("Selected classifier does not provide native probabilities")
        values = classifier.predict_proba(transformed)[0]
        return {
            str(label): float(probability)
            for label, probability in zip(classifier.classes_, values, strict=True)
        }

    def predict_proba(self, reviews: list[str]) -> list[dict[str, float]]:
        """Return native class probabilities for every input review."""
        classifier = self.pipeline.named_steps["classifier"]
        if not hasattr(classifier, "predict_proba"):
            raise RuntimeError("Selected classifier does not provide native probabilities")
        values = self.pipeline.predict_proba(reviews)
        classes = classifier.classes_.tolist()
        return [
            {label: float(probability) for label, probability in zip(classes, row, strict=True)}
            for row in values
        ]

    def _explain_transformed(
        self, transformed: Any, predicted: str, *, top_n: int = 5
    ) -> dict[str, Any]:
        """Rank active linear feature contributions for one transformed review."""
        classifier = self.pipeline.named_steps["classifier"]
        if not hasattr(classifier, "coef_"):
            raise RuntimeError("Feature contributions require a fitted linear classifier")
        if not sparse.issparse(transformed):
            transformed = sparse.csr_matrix(transformed)
        class_index = list(classifier.classes_).index(predicted)
        contributions = transformed.multiply(classifier.coef_[class_index]).tocsr()
        feature_names = getattr(self, "_feature_names_cache", None)
        if feature_names is None:
            feature_names = self.pipeline.named_steps["features"].get_feature_names_out()
            self._feature_names_cache = feature_names
        pairs = [
            (str(feature_names[index]), float(value))
            for index, value in zip(contributions.indices, contributions.data, strict=True)
        ]

        def item(feature: str, value: float) -> dict[str, Any]:
            """Convert one raw feature contribution into the API schema."""
            if feature.startswith("word__"):
                source, clean = "word_ngram", feature.removeprefix("word__")
            elif feature.startswith("char__"):
                source, clean = "character_ngram", feature.removeprefix("char__")
            else:
                source, clean = "word_ngram", feature
            return {"feature": clean, "source": source, "contribution": round(value, 6)}

        word_pairs = [pair for pair in pairs if not pair[0].startswith("char__")]
        char_pairs = [pair for pair in pairs if pair[0].startswith("char__")]
        supporting = sorted(word_pairs, key=lambda pair: pair[1], reverse=True)[:top_n]
        opposing = sorted(word_pairs, key=lambda pair: pair[1])[:top_n]
        technical = sorted(char_pairs, key=lambda pair: abs(pair[1]), reverse=True)[:top_n]
        return {
            "predicted_label": predicted,
            "supporting_word_features": [item(*pair) for pair in supporting if pair[1] > 0],
            "opposing_word_features": [item(*pair) for pair in opposing if pair[1] < 0],
            "technical_character_features": [item(*pair) for pair in technical],
            "limitation": "Linear feature contributions are associative, not causal explanations.",
        }

    def infer(self, review: str, *, explain: bool = False, top_n: int = 5) -> ModelInference:
        """Transform once, then derive label, probabilities, and optional contributions."""
        transformed = self.pipeline[:-1].transform([review])
        classifier = self.pipeline.named_steps["classifier"]
        label = self._validate_label(str(classifier.predict(transformed)[0]))
        probabilities = self._probability_dict(transformed)
        explanation = (
            self._explain_transformed(transformed, label, top_n=top_n) if explain else None
        )
        return ModelInference(label, probabilities, explanation)

    def explain(self, review: str, *, top_n: int = 5) -> dict[str, Any]:
        """Return active linear feature contributions, not causal explanations."""
        inference = self.infer(review, explain=True, top_n=top_n)
        if inference.explanation is None:
            raise RuntimeError("Explanation generation unexpectedly returned no result")
        return inference.explanation

    def save(self, path: str | Path) -> None:
        """Serialize the controlled model artifact with compression."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> SentimentModel:
        """Load and type-check a trusted local model artifact."""
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        loaded = joblib.load(model_path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact is not a {cls.__name__}: {model_path}")
        return loaded
