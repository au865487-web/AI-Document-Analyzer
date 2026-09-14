"""Exceptions for LLM answer generation."""


class GenerationError(Exception):
    """Base exception for all answer generation failures."""


class ConfigurationError(GenerationError):
    """Raised when Groq API key or required model settings are missing."""


class AuthenticationError(GenerationError):
    """Raised when the Groq API key is invalid or unauthorized."""


class RateLimitError(GenerationError):
    """Raised when the Groq rate limit is exceeded."""


class GenerationTimeoutError(GenerationError):
    """Raised when the Groq API request times out."""


class GenerationConnectionError(GenerationError):
    """Raised when network connectivity to the Groq API fails."""
