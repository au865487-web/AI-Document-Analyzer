"""Groq-powered LLM answer synthesis over a ContextPayload."""
from __future__ import annotations

import re
from typing import Any, NoReturn

from app import config
from app.context.models import ContextPayload, SourceItem
from app.generation.exceptions import (
    AuthenticationError,
    ConfigurationError,
    GenerationConnectionError,
    GenerationError,
    GenerationTimeoutError,
    RateLimitError,
)
from app.generation.models import Citation, GeneratedAnswer
from app.generation.prompts import SYSTEM_INSTRUCTIONS, build_user_message

_CITATION_RE = re.compile(r"\[Source (\d+)\]")

_EMPTY_CONTEXT_ANSWER = "No relevant documents were found to answer your question."


class AnswerGenerator:
    """Synthesize an answer from a ContextPayload using Groq chat completions.

    Groq client construction is lazy. An injected mock/stub client works
    fully offline and does not require GROQ_API_KEY.
    """

    def __init__(
        self,
        client: Any = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self._injected_client = client
        self._lazy_client: Any = None

        if model is not None:
            self.model = model
        elif config.GROQ_MODEL is not None and config.GROQ_MODEL != "":
            self.model = config.GROQ_MODEL
        else:
            self.model = config.DEFAULT_GROQ_MODEL

        if api_key is None:
            self._api_key = config.GROQ_API_KEY
        else:
            self._api_key = api_key

        self.temperature = (
            config.DEFAULT_TEMPERATURE if temperature is None else temperature
        )
        self.max_tokens = (
            config.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        )

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client

        if self._lazy_client is not None:
            return self._lazy_client

        if self._api_key is None or self._api_key == "":
            raise ConfigurationError(
                "GROQ_API_KEY is not configured. Set it in the environment "
                "or pass api_key explicitly."
            )

        from groq import Groq

        self._lazy_client = Groq(api_key=self._api_key)
        return self._lazy_client

    def generate(self, question: str, context: ContextPayload) -> GeneratedAnswer:
        """Generate an answer for ``question`` using ``context``."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if context is None:
            raise ValueError("context is required")

        formatted = context.formatted_context or ""
        if context.is_empty or not formatted.strip():
            return GeneratedAnswer(
                answer=_EMPTY_CONTEXT_ANSWER,
                citations=[],
                cited_sources=[],
                invalid_citations=[],
                model=self.model,
                is_empty_context=True,
                token_usage={},
            )

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {
                "role": "user",
                "content": build_user_message(question, formatted),
            },
        ]

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            self._raise_mapped(exc)

        answer_text = _extract_answer_text(response)
        citations, cited_sources, invalid_citations = _validate_citations(
            answer_text, context.sources_by_id
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            cited_sources=cited_sources,
            invalid_citations=invalid_citations,
            model=self.model,
            is_empty_context=False,
            token_usage=_extract_token_usage(response),
        )

    def _raise_mapped(self, exc: BaseException) -> NoReturn:
        """Map Groq SDK exceptions onto generation exceptions, then re-raise."""
        import groq

        if isinstance(exc, groq.AuthenticationError):
            raise AuthenticationError("Groq authentication failed.") from exc
        if isinstance(exc, groq.RateLimitError):
            raise RateLimitError("Groq rate limit exceeded.") from exc
        if isinstance(exc, groq.APITimeoutError):
            raise GenerationTimeoutError("Groq request timed out.") from exc
        if isinstance(exc, groq.APIConnectionError):
            raise GenerationConnectionError(
                "Failed to connect to the Groq API."
            ) from exc
        if isinstance(exc, groq.APIError):
            raise GenerationError("Groq API request failed.") from exc
        raise GenerationError("Unexpected error during answer generation.") from exc


def generate_answer(
    question: str,
    context: ContextPayload,
    *,
    client: Any = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GeneratedAnswer:
    """Convenience wrapper around :class:`AnswerGenerator`."""
    generator = AnswerGenerator(
        client=client,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return generator.generate(question, context)


def _extract_answer_text(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            raise GenerationError("Groq response contained no choices.")
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        if message is None:
            raise GenerationError("Groq response choice is missing a message.")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if content is None:
            raise GenerationError("Groq response message is missing content.")
        if not isinstance(content, str):
            raise GenerationError("Groq response content is not text.")
        return content
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError("Unexpected Groq response format.") from exc


def _extract_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {}

    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(key)
        if isinstance(value, int):
            result[key] = value
    return result


def _validate_citations(
    answer: str,
    sources_by_id: dict[int, SourceItem],
) -> tuple[list[Citation], list[SourceItem], list[str]]:
    citations: list[Citation] = []
    cited_sources: list[SourceItem] = []
    invalid_citations: list[str] = []
    seen: set[int] = set()

    for match in _CITATION_RE.finditer(answer):
        tag = match.group(0)
        source_id = int(match.group(1))
        if source_id in seen:
            continue
        seen.add(source_id)

        source = sources_by_id.get(source_id)
        if source is None:
            invalid_citations.append(tag)
            continue

        citations.append(
            Citation(
                source_id=source.source_id,
                document_id=source.document_id,
                page_number=source.page_number,
                chunk_index=source.chunk_index,
                text=source.text,
                citation_tag=source.citation_tag,
                citation_label=source.citation_label,
            )
        )
        cited_sources.append(source)

    return citations, cited_sources, invalid_citations
