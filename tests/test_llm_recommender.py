import hashlib
import json

import pytest

from dutch_sentiment.models.llm_advisor import (
    LLMRecommender,
    _build_system_prompt,
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
    result = recommender.recommend("Deze film was goed.")
    assert result.status == "unavailable"
    assert result.prompt_profile == "deterministic-24-shot-v1"
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


def test_runtime_uses_exact_frozen_24_shot_prompt() -> None:
    prompt = _build_system_prompt()
    examples = json.loads(prompt.split("training partition:\n", maxsplit=1)[1])

    assert len(examples) == 24
    assert {
        label: sum(item["label"] == label for item in examples)
        for label in {"Positive", "Average", "Negative"}
    } == {"Positive": 8, "Average": 8, "Negative": 8}
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "d4ca19fd4f4bb457a5d36ed4e90e1bcf925157a7f7f36496d3c8ab0c1fa0b908"
    )


def test_recommendation_sends_24_shot_prompt_and_bounded_output(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        """Return one valid DeepSeek-compatible response."""

        def raise_for_status(self) -> None:
            """Represent a successful response."""

        def json(self) -> dict[str, object]:
            """Return the minimal chat-completions response body."""
            return {
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": '{"label":"Positive"}'}}],
            }

    class FakeClient:
        """Capture the outbound request without network access."""

        def __init__(self, **_: object) -> None:
            """Accept the same keyword arguments as the HTTP client."""

        def __enter__(self) -> "FakeClient":
            """Enter the fake client context."""
            return self

        def __exit__(self, *_: object) -> None:
            """Exit the fake client context."""

        def post(self, path: str, *, json: dict[str, object]) -> FakeResponse:
            """Capture the request path and JSON payload."""
            captured.update(path=path, payload=json)
            return FakeResponse()

    monkeypatch.setattr("dutch_sentiment.models.llm_advisor.httpx.Client", FakeClient)
    result = LLMRecommender(api_key="secret").recommend("Deze film was uitstekend.")

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert captured["path"] == "/chat/completions"
    assert payload["max_tokens"] == 30
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["messages"][0]["content"] == _build_system_prompt()
    assert payload["messages"][1]["content"] == json.dumps(
        {"review": "Deze film was uitstekend."}, ensure_ascii=False
    )
    assert result.status == "ok"
    assert result.prompt_profile == "deterministic-24-shot-v1"


def test_load_api_key_from_file_ignores_missing_or_blank_files(tmp_path) -> None:
    assert _load_api_key_from_file(str(tmp_path / "missing")) is None
    blank = tmp_path / "blank"
    blank.write_text("\n", encoding="utf-8")
    assert _load_api_key_from_file(str(blank)) is None


def test_llm_recommendation_parser_accepts_strict_supported_label() -> None:
    assert _parse_recommendation('{"label":"Negative"}') == "Negative"


@pytest.mark.parametrize(
    "content",
    [
        "Positive",
        '{"label":"Neutral"}',
        '{"rationale":"missing label"}',
        '{"label":"Positive","confidence":0.9}',
    ],
)
def test_llm_recommendation_parser_rejects_invalid_content(content: str) -> None:
    with pytest.raises(ValueError):
        _parse_recommendation(content)
