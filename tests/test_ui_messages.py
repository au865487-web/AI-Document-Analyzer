"""Tests for app.ui.messages — no Streamlit imports."""
from app.generation.exceptions import (
    AuthenticationError,
    ConfigurationError,
    GenerationConnectionError,
    GenerationError,
    GenerationTimeoutError,
    RateLimitError,
)
from app.ingestion.extractor import ExtractionError
from app.pipeline import AskResult, IndexResult
from app.ui.messages import (
    EMPTY_INDEX,
    EMPTY_RETRIEVAL,
    empty_ask_status,
    index_success,
    user_message_for_exception,
)


def _ask_result(*, empty_index=False, empty_retrieval=False):
    from app.context.models import ContextPayload
    from app.generation.models import GeneratedAnswer

    return AskResult(
        answer=GeneratedAnswer(
            answer="x",
            citations=[],
            cited_sources=[],
            invalid_citations=[],
            model="test",
            is_empty_context=empty_retrieval,
        ),
        context=ContextPayload(
            formatted_context="",
            sources=[],
            sources_by_id={},
            is_empty=empty_retrieval,
        ),
        retrieval_count=0,
        is_empty_index=empty_index,
        is_empty_retrieval=empty_retrieval,
    )


def test_index_success_plural_and_zero():
    assert "2 chunks" in index_success(IndexResult("a.pdf", 2))
    assert "1 chunk." in index_success(IndexResult("a.pdf", 1))
    assert "No searchable text" in index_success(IndexResult("empty.txt", 0))


def test_empty_ask_status():
    assert empty_ask_status(_ask_result(empty_index=True)) == EMPTY_INDEX
    assert empty_ask_status(_ask_result(empty_retrieval=True)) == EMPTY_RETRIEVAL
    from app.context.models import ContextPayload
    from app.generation.models import GeneratedAnswer

    filled = AskResult(
        answer=GeneratedAnswer(
            answer="ok",
            citations=[],
            cited_sources=[],
            invalid_citations=[],
            model="test",
        ),
        context=ContextPayload(formatted_context="ctx", sources=[], sources_by_id={}),
        retrieval_count=1,
        is_empty_index=False,
        is_empty_retrieval=False,
    )
    assert empty_ask_status(filled) is None


def test_exception_messages_are_safe():
    mapping = [
        (ExtractionError("File not found: secret.pdf"), "File not found"),
        (ConfigurationError("internal"), "not configured"),
        (AuthenticationError("internal"), "rejected"),
        (RateLimitError("internal"), "rate limit"),
        (GenerationTimeoutError("internal"), "timed out"),
        (GenerationConnectionError("internal"), "Could not connect"),
        (GenerationError("internal"), "Answer generation failed"),
        (ValueError("question must be a non-empty string"), "question must"),
    ]
    for exc, snippet in mapping:
        message = user_message_for_exception(exc)
        assert snippet.lower() in message.lower()
        assert "GROQ_API_KEY=" not in message
        assert "sk-" not in message


def test_unknown_exception_is_generic():
    message = user_message_for_exception(RuntimeError("disk key=abc"))
    assert message == "Something went wrong. Please try again."
    assert "abc" not in message
