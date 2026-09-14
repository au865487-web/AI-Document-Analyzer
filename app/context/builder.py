"""Build structured, citation-ready context from retrieved search results."""
from app import config
from app.context.models import ContextPayload, SourceItem
from app.storage.vector_store import SearchResult

SEPARATOR = "\n\n---\n\n"


def _format_source_block(source_id: int, result: SearchResult) -> str:
    lines = [f"[Source {source_id}]", f"Document: {result.document_id}"]
    if result.page_number is not None:
        lines.append(f"Page: {result.page_number}")
    lines.append("Content:")
    lines.append(result.text.strip())
    return "\n".join(lines)


class ContextBuilder:
    """Assembles retrieved SearchResults into deterministic, citation-ready context."""

    def __init__(
        self,
        max_context_chars: int | None = None,
        max_sources: int | None = None,
    ):
        self.max_context_chars = (
            config.MAX_CONTEXT_CHARS
            if max_context_chars is None
            else max_context_chars
        )
        self.max_sources = (
            config.MAX_SOURCES if max_sources is None else max_sources
        )

        if self.max_context_chars < 0:
            raise ValueError("max_context_chars must be non-negative")
        if self.max_sources < 0:
            raise ValueError("max_sources must be non-negative")

    def build(self, results: list[SearchResult]) -> ContextPayload:
        """Process retrieved results into a ContextPayload.

        1. Filters out empty/whitespace-only results.
        2. Deduplicates chunks by (document_id, chunk_index) while preserving rank.
        3. Clamps count to max_sources.
        4. Accumulates chunks within max_context_chars (evaluated against the actual
           formatted context, including tags, metadata, separators, and content).
        5. Assigns fresh, sequential source_ids (1, 2, ...) for each build call.
        6. Formats deterministic string blocks and constructs ContextPayload.
        """
        if not results:
            return ContextPayload(
                formatted_context="",
                sources=[],
                sources_by_id={},
                total_characters=0,
                truncated=False,
                is_empty=True,
            )

        # 1. Deduplicate by (document_id, chunk_index) while preserving rank order
        seen_keys = set()
        unique_candidates: list[SearchResult] = []

        for r in results:
            if not r.text or not r.text.strip():
                continue
            key = (r.document_id, r.chunk_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_candidates.append(r)

        if not unique_candidates:
            return ContextPayload(
                formatted_context="",
                sources=[],
                sources_by_id={},
                total_characters=0,
                truncated=False,
                is_empty=True,
            )

        # 2. Limit candidate count to max_sources
        truncated = False
        if self.max_sources < len(unique_candidates):
            candidates_to_process = unique_candidates[: self.max_sources]
            truncated = True
        else:
            candidates_to_process = unique_candidates

        # 3. Format blocks incrementally within max_context_chars
        current_blocks: list[str] = []
        sources: list[SourceItem] = []
        sources_by_id: dict[int, SourceItem] = {}
        current_len = 0

        for r in candidates_to_process:
            next_source_id = len(current_blocks) + 1
            block = _format_source_block(next_source_id, r)

            sep_len = len(SEPARATOR) if current_blocks else 0
            projected_len = current_len + sep_len + len(block)

            if projected_len > self.max_context_chars:
                truncated = True
                break

            current_blocks.append(block)
            current_len = projected_len

            source_item = SourceItem(
                source_id=next_source_id,
                document_id=r.document_id,
                chunk_index=r.chunk_index,
                text=r.text.strip(),
                page_number=r.page_number,
                score=r.score,
                distance=r.distance,
            )
            sources.append(source_item)
            sources_by_id[next_source_id] = source_item

        formatted_context = SEPARATOR.join(current_blocks)

        return ContextPayload(
            formatted_context=formatted_context,
            sources=sources,
            sources_by_id=sources_by_id,
            total_characters=len(formatted_context),
            truncated=truncated,
            is_empty=len(sources) == 0,
        )


def build_context(
    results: list[SearchResult],
    max_context_chars: int | None = None,
    max_sources: int | None = None,
) -> ContextPayload:
    """Assemble retrieved SearchResults into a ContextPayload."""
    builder = ContextBuilder(
        max_context_chars=max_context_chars,
        max_sources=max_sources,
    )
    return builder.build(results)
