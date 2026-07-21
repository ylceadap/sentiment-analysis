from pathlib import Path

from dutch_sentiment.constants import LABELS
from dutch_sentiment.models.classical import ModelSpec, SentimentModel, build_pipeline


def _model() -> SentimentModel:
    config = {
        "min_df": 1,
        "max_df": 1.0,
        "word_max_features": 500,
        "char_max_features": 500,
        "max_iter": 500,
        "random_seed": 11,
    }
    reviews = [
        "prachtige film en geweldig gespeeld",
        "fantastisch verhaal en heel mooi",
        "goede acteurs en een sterk einde",
        "de film was gewoon gemiddeld",
        "redelijk maar niet bijzonder",
        "sommige scènes goed andere matig",
        "verschrikkelijke film en slecht gespeeld",
        "saai verhaal en een waardeloos einde",
        "heel slecht en absoluut niet boeiend",
    ]
    labels = ["Positive"] * 3 + ["Average"] * 3 + ["Negative"] * 3
    return SentimentModel(
        build_pipeline(ModelSpec("test", "combined", "balanced"), config),
        version="test-v1",
    ).fit(reviews, labels)


def test_model_fit_predict_probability_and_explanation_contract() -> None:
    model = _model()
    review = "Een prachtige film met goede acteurs"
    prediction = model.predict([review])[0]
    assert prediction in LABELS
    probabilities = model.predict_proba([review])[0]
    assert set(probabilities) == set(LABELS)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-5
    explanation = model.explain(review)
    assert explanation["predicted_label"] == prediction
    assert explanation["supporting_word_features"]
    assert all(item["source"] == "word_ngram" for item in explanation["supporting_word_features"])
    inference = model.infer(review, explain=True)
    assert inference.label == prediction
    assert inference.probabilities == probabilities
    assert inference.explanation == explanation
    assert model._feature_names_cache is not None


def test_model_is_deterministic_and_survives_round_trip(tmp_path: Path) -> None:
    model = _model()
    reviews = ["Dit was een goede en mooie film", "Dit was saai en erg slecht"]
    before = model.predict(reviews)
    assert before == model.predict(reviews)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = SentimentModel.load(path)
    assert loaded.version == "test-v1"
    assert loaded.predict(reviews) == before
