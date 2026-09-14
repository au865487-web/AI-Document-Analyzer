"""Data models for generated answers and verified citations."""
from dataclasses import dataclass, field

from app.context.models import SourceItem


@dataclass(frozen=True)
class Citation:
    """A citation verified against ContextPayload.sources_by_id.

    Metadata is copied from the matching SourceItem. The LLM response
    is never used as the source of document/page/chunk fields.
    """

    source_id: int
    document_id: str
    page_number: int | None
    chunk_index: int
    text: str
    citation_tag: str
    citation_label: str


@dataclass(frozen=True)
class GeneratedAnswer:
    """LLM-synthesized answer with whitelist-validated citations."""

    answer: str
    citations: list[Citation]
    cited_sources: list[SourceItem]
    invalid_citations: list[str]
    model: str
    is_empty_context: bool = False
    token_usage: dict[str, int] = field(default_factory=dict)
