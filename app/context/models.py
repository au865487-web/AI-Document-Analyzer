"""Data models for structured context and citations."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceItem:
    """An individual source chunk incorporated into the LLM context.

    Attributes:
        source_id: 1-based sequential integer (1, 2, 3...) assigned within a context payload.
        document_id: Origin document identifier (e.g. "report.pdf" or "notes").
        chunk_index: 0-based position of the chunk in the document.
        text: Text content of the chunk.
        page_number: 1-based page number for PDFs, or None for unpaginated sources (e.g. TXT).
        score: Retrieval similarity score (cosine similarity, higher is better).
        distance: Retrieval distance (cosine distance, lower is better).
    """

    source_id: int
    document_id: str
    chunk_index: int
    text: str
    page_number: int | None
    score: float
    distance: float

    @property
    def citation_tag(self) -> str:
        """The inline citation tag the LLM is instructed to output (e.g. '[Source 1]')."""
        return f"[Source {self.source_id}]"

    @property
    def citation_label(self) -> str:
        """Human-readable citation label for UI display."""
        if self.page_number is not None:
            return f"{self.document_id} (Page {self.page_number})"
        return f"{self.document_id}"


@dataclass(frozen=True)
class ContextPayload:
    """Complete structured context ready for LLM prompt injection.

    Attributes:
        formatted_context: The final prompt-ready string block.
        sources: Ordered list of sources included in formatted_context.
        sources_by_id: Map of source_id -> SourceItem for O(1) citation validation.
        total_characters: Total character length of formatted_context (matching the safety budget).
        truncated: True if any candidate search results were excluded due to budget or limits.
        is_empty: True if no valid sources could be incorporated.
    """

    formatted_context: str
    sources: list[SourceItem]
    sources_by_id: dict[int, SourceItem] = field(default_factory=dict)
    total_characters: int = 0
    truncated: bool = False
    is_empty: bool = False
