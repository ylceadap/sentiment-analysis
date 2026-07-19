import pytest

from dutch_sentiment.llm_recommender import (
    LLMRecommender,
    _load_api_key_from_file,
    _parse_recommendation,
)


def test_llm_recommender_is_unavailable_without_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_FILE", raising=False)
    monkeypatch.delenv("LLM_API_KEY_FILE", raising=False)
    recommender = LLMRecommender.from_environment()
    result = recommender.recommend("Deze film was goed.", detected_language="dutch")
    assert result.status == "unavailable"
    assert result.label is None
    assert "API_KEY" in str(result.warning)


def test_llm_recommender_loads_default_ignored_key_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_FILE", raising=False)
    monkeypatch.delenv("LLM_API_KEY_FILE", raising=False)
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir()
    (secrets_dir / "deepseek_api_key").write_text("file-secret\n", encoding="utf-8")
    assert LLMRecommender.from_environment().api_key == "file-secret"


def test_llm_recommender_loads_configured_key_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    key_file = tmp_path / "llm_key"
    key_file.write_text("configured-secret\n", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY_FILE", str(key_file))
    assert LLMRecommender.from_environment().api_key == "configured-secret"


def test_environment_key_takes_precedence_over_file(tmp_path, monkeypatch) -> None:
    key_file = tmp_path / "llm_key"
    key_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-secret")
    monkeypatch.setenv("LLM_API_KEY_FILE", str(key_file))
    assert LLMRecommender.from_environment().api_key == "env-secret"


def test_load_api_key_from_file_ignores_missing_or_blank_files(tmp_path) -> None:
    assert _load_api_key_from_file(str(tmp_path / "missing")) is None
    blank = tmp_path / "blank"
    blank.write_text("\n", encoding="utf-8")
    assert _load_api_key_from_file(str(blank)) is None


def test_llm_recommendation_parser_accepts_strict_supported_label() -> None:
    label, rationale, confidence = _parse_recommendation(
        '{"label":"Negative","rationale":"The review is clearly unfavorable.","confidence":0.86}'
    )
    assert label == "Negative"
    assert rationale == "The review is clearly unfavorable."
    assert confidence == 0.86


@pytest.mark.parametrize(
    "content",
    [
        "Positive",
        '{"label":"Neutral"}',
        '{"rationale":"missing label"}',
    ],
)
def test_llm_recommendation_parser_rejects_invalid_content(content: str) -> None:
    with pytest.raises(ValueError):
        _parse_recommendation(content)
