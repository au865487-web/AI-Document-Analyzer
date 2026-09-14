"""Semantic retrieval over chunked documents.

The retriever ties the application pieces together: it indexes documents
by chunking their text and embedding the chunks, and it answers natural
language queries by embedding the query and searching the vector store.
It contains no FAISS-specific logic — that stays behind
:class:`app.storage.vector_store.VectorStore` — and embeds both documents
and queries with the same :class:`app.embeddings.embedder.Embedder`.
"""
from pathlib import Path

from app.chunking.chunker import chunk_pages, chunk_text
from app.cleaning.cleaner import clean_text
from app.embeddings.embedder import Embedder
from app.ingestion.extractor import ExtractedPage, extract_pages
from app.storage.vector_store import SearchResult, VectorStore

DEFAULT_TOP_K = 10


class Retriever:
    """Index document chunks and search them semantically."""

    def __init__(self, store=None, embedder=None, chunker=chunk_text):
        self._store = store or VectorStore()
        self._embedder = embedder or Embedder()
        self._chunker = chunker

    def add_document(self, document_id: str, text: str) -> int:
        """Chunk, embed, and store ``text`` under ``document_id``.

        Re-indexing a document id replaces its previous chunks rather than
        duplicating them. Returns the number of chunks stored (0 for
        empty input).
        """
        chunks = self._chunker(text, document_id=document_id)
        if not chunks:
            return 0
        vectors = self._embedder.embed_documents([c.text for c in chunks])
        self._store.delete_document(document_id)
        return self._store.upsert_chunks(chunks, vectors)

    def add_file(self, path, document_id: str | None = None) -> int:
        """Ingest a PDF or TXT file end-to-end: extract, clean, chunk, embed, and store.

        Preserves page-level metadata (1-based page number for PDFs, None for TXT).
        If ``document_id`` is not provided, defaults to the file stem.

        Replaces any existing chunks stored under ``document_id`` only after
        extraction, chunking, and embedding successfully complete, preventing
        partial or corrupted index states.

        Returns:
            The number of chunks stored (0 if the document yields no text).

        Raises:
            ExtractionError: if the file cannot be read, decoded, or extracted.
        """
        path = Path(path)
        doc_id = document_id or path.stem

        # 1. Extraction (fails fast without modifying store)
        pages = extract_pages(path)

        # 2. Cleaning (normalize text per page, discard blank pages)
        cleaned_pages = []
        for p in pages:
            cleaned_str = clean_text(p.text)
            if cleaned_str.strip():
                cleaned_pages.append(
                    ExtractedPage(page_number=p.page_number, text=cleaned_str)
                )

        if not cleaned_pages:
            return 0

        # 3. Chunking (maintains page_number provenance and continuous chunk_index)
        chunks = chunk_pages(cleaned_pages, document_id=doc_id)
        if not chunks:
            return 0

        # 4. Embedding (computed before touching the store)
        vectors = self._embedder.embed_documents([c.text for c in chunks])

        # 5. Replacement (delete prior chunks and upsert new ones)
        self._store.delete_document(doc_id)
        return self._store.upsert_chunks(chunks, vectors)

    def delete_document(self, document_id: str) -> int:
        """Delete every chunk belonging to ``document_id``; return how many."""
        return self._store.delete_document(document_id)

    def clear(self) -> int:
        """Remove every chunk from the store; return how many were removed."""
        return self._store.clear()

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self._store.count()

    def list_documents(self) -> list[tuple[str, int]]:
        """Return unique document IDs and indexed chunk counts from the store."""
        return self._store.list_documents()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """Embed ``query`` and return the ``top_k`` most similar chunks.

        A blank query or an empty collection returns ``[]``. Results are
        ordered by distance (most similar first) and include the chunk's
        text, document id, chunk index, distance, and similarity score.

        Raises:
            ValueError: if ``top_k`` is not a positive integer.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(query, str) or not query.strip():
            return []
        if self._store.count() == 0:
            return []

        query_vector = self._embedder.embed_query(query)
        return self._store.query(query_vector, top_k)
