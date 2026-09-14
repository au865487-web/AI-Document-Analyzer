"""Offline tests for app.generation.generator.

These tests use a protocol-compatible stub client. They must never make
real Groq API calls or require GROQ_API_KEY.
"""
from types import SimpleNamespace

import httpx
import pytest

from app import config
from app.context.models import ContextPayload, SourceItem
from app.generation import (
    AnswerGenerator,
    AuthenticationError,
    ConfigurationError,
    GenerationConnectionError,
    GenerationTimeoutError,
    RateLimitError,
    generate_answer,
)
from app.generation.exceptions import GenerationError
from app.generation.prompts import build_user_message


class _FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.completions = _FakeCompletions(response=response, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def _httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _httpx_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_httpx_request())


def _make_source(
    source_id: int,
    *,
    document_id: str = "report.pdf",
    chunk_index: int = 0,
    text: str = "Operating profit rose 4.2 percent.",
    page_number: int | None = 3,
) -> SourceItem:
    return SourceItem(
        source_id=source_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        page_number=page_number,
        score=0.9,
        distance=0.1,
    )


def _make_context(sources: list[SourceItem], *, is_empty: bool = False) -> ContextPayload:
    if is_empty or not sources:
        return ContextPayload(
            formatted_context="",
            sources=[],
            sources_by_id={},
            total_characters=0,
            truncated=False,
            is_empty=True,
        )
    blocks = []
    for src in sources:
        lines = [src.citation_tag, f"Document: {src.document_id}"]
        if src.page_number is not None:
            lines.append(f"Page: {src.page_number}")
        lines.append("Content:")
        lines.append(src.text)
        blocks.append("\n".join(lines))
    formatted = "\n\n---\n\n".join(blocks)
    return ContextPayload(
        formatted_context=formatted,
        sources=list(sources),
        sources_by_id={s.source_id: s for s in sources},
        total_characters=len(formatted),
        truncated=False,
        is_empty=False,
    )


def _completion(content: str, usage=None):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_mock_client_works_without_groq_api_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    source = _make_source(1)
    context = _make_context([source])
    client = _FakeClient(
        response=_completion("Profit rose. [Source 1]")
    )
    result = AnswerGenerator(client=client).generate("What happened?", context)
    assert "Profit rose" in result.answer
    assert len(client.completions.calls) == 1


def test_importing_generation_does_not_instantiate_groq(monkeypatch):
    import groq

    def _boom(*_args, **_kwargs):
        raise AssertionError("Groq client must not be instantiated on import")

    monkeypatch.setattr(groq, "Groq", _boom)
    import importlib

    import app.generation
    import app.generation.generator as generator_mod

    importlib.reload(generator_mod)
    importlib.reload(app.generation)


def test_missing_api_key_without_injected_client_raises(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    generator = AnswerGenerator()
    context = _make_context([_make_source(1)])
    with pytest.raises(ConfigurationError):
        generator.generate("What happened?", context)


def test_basic_answer_with_source_1():
    source = _make_source(1)
    context = _make_context([source])
    client = _FakeClient(response=_completion("The margin rose. [Source 1]"))
    result = AnswerGenerator(client=client).generate("What is the margin?", context)
    assert result.answer == "The margin rose. [Source 1]"
    assert len(result.citations) == 1
    assert result.citations[0].source_id == 1
    assert result.is_empty_context is False


def test_citation_metadata_comes_from_source_item():
    source = _make_source(
        1,
        document_id="notes.txt",
        chunk_index=4,
        text="Cluster node synchronized at 12:00 UTC.",
        page_number=None,
    )
    context = _make_context([source])
    client = _FakeClient(
        response=_completion(
            "The node synchronized at 12:00 UTC. [Source 1]"
        )
    )
    result = AnswerGenerator(client=client).generate("When did it sync?", context)
    citation = result.citations[0]
    assert citation.document_id == "notes.txt"
    assert citation.page_number is None
    assert citation.chunk_index == 4
    assert citation.text == "Cluster node synchronized at 12:00 UTC."
    assert citation.citation_tag == source.citation_tag
    assert citation.citation_label == source.citation_label
    assert result.cited_sources == [source]


def test_multiple_citations_preserve_first_appearance_order():
    sources = [
        _make_source(1, document_id="a.pdf", text="Alpha fact.", page_number=1),
        _make_source(2, document_id="b.pdf", chunk_index=1, text="Beta fact.", page_number=2),
        _make_source(3, document_id="c.pdf", chunk_index=2, text="Gamma fact.", page_number=3),
    ]
    context = _make_context(sources)
    client = _FakeClient(
        response=_completion("See [Source 3] then [Source 1] and [Source 2].")
    )
    result = AnswerGenerator(client=client).generate("Summarize.", context)
    assert [c.source_id for c in result.citations] == [3, 1, 2]
    assert [s.source_id for s in result.cited_sources] == [3, 1, 2]


def test_hallucinated_source_is_invalid():
    context = _make_context([_make_source(1)])
    client = _FakeClient(
        response=_completion("A claim with [Source 1] and a fake [Source 99].")
    )
    result = AnswerGenerator(client=client).generate("What happened?", context)
    assert [c.source_id for c in result.citations] == [1]
    assert result.invalid_citations == ["[Source 99]"]


def test_empty_context_bypasses_api():
    client = _FakeClient(response=_completion("should not be used"))
    result = AnswerGenerator(client=client, model="test-model").generate(
        "What happened?",
        _make_context([], is_empty=True),
    )
    assert result.is_empty_context is True
    assert result.answer == "No relevant documents were found to answer your question."
    assert result.citations == []
    assert result.cited_sources == []
    assert result.invalid_citations == []
    assert result.model == "test-model"
    assert client.completions.calls == []


def test_authentication_error_mapping():
    import groq

    err = groq.AuthenticationError(
        "invalid api key",
        response=_httpx_response(401),
        body=None,
    )
    client = _FakeClient(error=err)
    with pytest.raises(AuthenticationError) as exc_info:
        AnswerGenerator(client=client).generate(
            "What happened?", _make_context([_make_source(1)])
        )
    assert exc_info.value.__cause__ is err


def test_rate_limit_error_mapping():
    import groq

    err = groq.RateLimitError(
        "rate limited",
        response=_httpx_response(429),
        body=None,
    )
    client = _FakeClient(error=err)
    with pytest.raises(RateLimitError) as exc_info:
        AnswerGenerator(client=client).generate(
            "What happened?", _make_context([_make_source(1)])
        )
    assert exc_info.value.__cause__ is err


def test_timeout_error_mapping():
    import groq

    err = groq.APITimeoutError(_httpx_request())
    client = _FakeClient(error=err)
    with pytest.raises(GenerationTimeoutError) as exc_info:
        AnswerGenerator(client=client).generate(
            "What happened?", _make_context([_make_source(1)])
        )
    assert exc_info.value.__cause__ is err


def test_connection_error_mapping():
    import groq

    err = groq.APIConnectionError(request=_httpx_request())
    client = _FakeClient(error=err)
    with pytest.raises(GenerationConnectionError) as exc_info:
        AnswerGenerator(client=client).generate(
            "What happened?", _make_context([_make_source(1)])
        )
    assert exc_info.value.__cause__ is err


def test_explicit_temperature_and_model_are_preserved():
    client = _FakeClient(response=_completion("ok [Source 1]"))
    generator = AnswerGenerator(
        client=client,
        model="explicit-model",
        temperature=0.25,
        max_tokens=64,
    )
    assert generator.model == "explicit-model"
    assert generator.temperature == 0.25
    assert generator.max_tokens == 64
    result = generator.generate("What happened?", _make_context([_make_source(1)]))
    assert result.model == "explicit-model"


def test_repeated_citation_ids_are_deduplicated():
    context = _make_context([_make_source(1), _make_source(2, chunk_index=1, text="Other.")])
    client = _FakeClient(
        response=_completion("First [Source 1] then again [Source 1] and [Source 2].")
    )
    result = AnswerGenerator(client=client).generate("What happened?", context)
    assert [c.source_id for c in result.citations] == [1, 2]


def test_token_usage_is_captured_when_available():
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    client = _FakeClient(response=_completion("ok [Source 1]", usage=usage))
    result = AnswerGenerator(client=client).generate(
        "What happened?", _make_context([_make_source(1)])
    )
    assert result.token_usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


def test_prompt_contains_context_and_question():
    source = _make_source(1, text="Unique context blob 42.")
    context = _make_context([source])
    question = "What is unique about this blob?"
    client = _FakeClient(response=_completion("It is unique. [Source 1]"))
    AnswerGenerator(client=client).generate(question, context)
    messages = client.completions.calls[0]["messages"]
    user_content = messages[1]["content"]
    assert "Unique context blob 42." in user_content
    assert question in user_content
    assert user_content == build_user_message(question, context.formatted_context)
    assert messages[0]["role"] == "system"


def test_api_call_receives_configured_model_temperature_max_tokens():
    client = _FakeClient(response=_completion("ok [Source 1]"))
    AnswerGenerator(
        client=client,
        model="configured-model",
        temperature=0.0,
        max_tokens=256,
    ).generate("What happened?", _make_context([_make_source(1)]))
    kwargs = client.completions.calls[0]
    assert kwargs["model"] == "configured-model"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 256


def test_generate_answer_helper_uses_injected_client():
    client = _FakeClient(response=_completion("helper [Source 1]"))
    result = generate_answer(
        "What happened?",
        _make_context([_make_source(1)]),
        client=client,
    )
    assert result.answer == "helper [Source 1]"


def test_malformed_response_raises_generation_error():
    client = _FakeClient(response=SimpleNamespace(choices=[]))
    with pytest.raises(GenerationError):
        AnswerGenerator(client=client).generate(
            "What happened?", _make_context([_make_source(1)])
        )


def test_model_resolution_uses_config_then_default(monkeypatch):
    monkeypatch.setattr(config, "GROQ_MODEL", "from-config")
    assert AnswerGenerator(client=_FakeClient()).model == "from-config"

    monkeypatch.setattr(config, "GROQ_MODEL", "")
    assert AnswerGenerator(client=_FakeClient()).model == config.DEFAULT_GROQ_MODEL

    monkeypatch.setattr(config, "GROQ_MODEL", "from-config")
    assert (
        AnswerGenerator(client=_FakeClient(), model="explicit").model == "explicit"
    )
