"""Split cleaned document text into deterministic, metadata-aware chunks.

Chunking is deliberately simple: whitespace runs are collapsed, sentence
boundaries are preferred as cut points (falling back to word and then
hard cuts), overlap bleeds a configurable tail into the next chunk, and
no empty chunks are produced. Every chunk carries the source document
id and its 0-based position so results can be traced back to the
original document.
"""
import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

_SENTENCE_ENDINGS = frozenset(".!?")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    """A single chunk of a document, tagged with provenance metadata."""

    document_id: str
    chunk_index: int
    text: str
    page_number: int | None = None


def chunk_text(
    text,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    document_id: str = "",
    page_number: int | None = None,
    start_index: int = 0,
) -> list[Chunk]:
    """Split ``text`` into a list of :class:`Chunk`.

    Args:
        text: Source text. Runs of whitespace are collapsed to single
            spaces before splitting.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of trailing characters shared with the following
            chunk (0 disables overlap). Must be smaller than ``chunk_size``.
        document_id: Identifier attached to every chunk produced.
        page_number: 1-based page number or None for unpaginated text.
        start_index: Starting value for ``chunk_index`` (defaults to 0).

    Returns:
        Chunks with ``chunk_index`` starting at ``start_index``. Empty input (including
        whitespace-only input) yields an empty list.

    Raises:
        ValueError: if ``chunk_size`` is not positive, ``overlap`` is
            negative, or ``overlap`` is not smaller than ``chunk_size``.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = _WHITESPACE.sub(" ", text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Chunk(document_id, start_index, text, page_number=page_number)]

    chunks = []
    start = 0
    index = start_index
    while start < len(text):
        end = _find_break(text, start, chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(document_id, index, piece, page_number=page_number))
            index += 1
        if end >= len(text):
            break
        # If only a short tail is left, take it as one final, bounded chunk
        # instead of letting overlap re-slice it into tiny fragments.
        if len(text) - end <= chunk_size - overlap:
            tail = text[end - overlap :].strip()
            if tail:
                chunks.append(Chunk(document_id, index, tail, page_number=page_number))
            break
        # Guarantee meaningful forward progress: if _find_break() returned a
        # position so close to `start` that `end - overlap` would be <= start
        # (i.e. the break point fell inside the overlap window), fall back to
        # using `end` as the next start.  This sacrifices overlap for that one
        # step but prevents the 1-character sliding-chunk bug.  Normal overlap
        # is fully preserved whenever end > start + overlap.
        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks


def chunk_pages(
    pages,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    document_id: str = "",
) -> list[Chunk]:
    """Split a sequence of pages into a unified list of :class:`Chunk`.

    Maintains a continuous 0-based ``chunk_index`` across all pages while
    preserving the ``page_number`` of each chunk's origin page.
    """
    all_chunks = []
    current_index = 0
    for page in pages:
        page_chunks = chunk_text(
            page.text,
            chunk_size=chunk_size,
            overlap=overlap,
            document_id=document_id,
            page_number=page.page_number,
            start_index=current_index,
        )
        all_chunks.extend(page_chunks)
        current_index += len(page_chunks)
    return all_chunks


def _find_break(text: str, start: int, chunk_size: int) -> int:
    """Find the best cut position within ``text[start:start + chunk_size]``.

    Prefers the last sentence ending (``.`` ``!`` ``?``) inside the window,
    then the last space, and finally hard-cuts at the window edge so the
    algorithm always makes progress (e.g. for very long words).
    """
    limit = min(start + chunk_size, len(text))

    for i in range(limit - 1, start, -1):
        if text[i] in _SENTENCE_ENDINGS:
            return _skip_spaces(text, i + 1, limit)
    for i in range(limit - 1, start, -1):
        if text[i] == " ":
            return _skip_spaces(text, i + 1, limit)
    return limit


def _skip_spaces(text: str, index: int, limit: int) -> int:
    """Advance past spaces so chunks do not start or end with whitespace."""
    while index < limit and text[index] == " ":
        index += 1
    return index