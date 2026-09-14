"""User-facing status and error copy for the Streamlit UI.

Kept free of Streamlit imports so messages can be unit-tested offline.
Never interpolates API keys into returned strings.
"""
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

EMPTY_INDEX = (
    "No documents are indexed yet. Upload a PDF or TXT file to get started."
)
EMPTY_RETRIEVAL = (
    "No relevant documents were found to answer your question."
)
UNSUPPORTED_UPLOAD = "Please upload a PDF or TXT file."
MISSING_QUESTION = "Enter a question before asking."
GROQ_READY = "Groq API key: configured"
GROQ_MISSING = (
    "Groq API key: not configured. Add GROQ_API_KEY to your .env file "
    "before asking questions. Indexing still works without it."
)


def index_success(result: IndexResult) -> str:
    """Describe a successful index operation."""
    if result.chunk_count == 0:
        return (
            f"No searchable text was indexed from '{result.document_id}'."
        )
    unit = "chunk" if result.chunk_count == 1 else "chunks"
    return f"Indexed '{result.document_id}' \u2014 {result.chunk_count} {unit}."


def empty_ask_status(result: AskResult) -> str | None:
    """Return an empty-state note for Ask, or None when results exist."""
    if result.is_empty_index:
        return EMPTY_INDEX
    if result.is_empty_retrieval:
        return EMPTY_RETRIEVAL
    return None


def user_message_for_exception(exc: BaseException) -> str:
    """Map an exception to a safe, user-visible error string."""
    if isinstance(exc, ExtractionError):
        return str(exc)
    if isinstance(exc, ConfigurationError):
        return (
            "The Groq API key is not configured. Add GROQ_API_KEY to your "
            ".env file and restart the app."
        )
    if isinstance(exc, AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY in your .env file."
    if isinstance(exc, RateLimitError):
        return "The Groq API rate limit was exceeded. Wait a moment and try again."
    if isinstance(exc, GenerationTimeoutError):
        return "The Groq request timed out. Try again."
    if isinstance(exc, GenerationConnectionError):
        return "Could not connect to the Groq API. Check your network and try again."
    if isinstance(exc, GenerationError):
        return "Answer generation failed. Try again, or check the Groq service status."
    if isinstance(exc, ValueError):
        text = str(exc).strip() or "Invalid input."
        return text
    return "Something went wrong. Please try again."
