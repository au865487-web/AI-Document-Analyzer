"""UI-agnostic orchestration over retrieval, context, and generation.

Streamlit (and tests) should call this module instead of reimplementing
search → context → answer wiring.
"""
from dataclasses import dataclass
from pathlib import Path

from app import config
from app.context.builder import build_context
from app.context.models import ContextPayload
from app.generation.generator import AnswerGenerator
from app.generation.models import GeneratedAnswer
from app.retrieval.retriever import DEFAULT_TOP_K, Retriever


@dataclass(frozen=True)
class IndexedDocument:
    """A document currently present in the vector index."""

    document_id: str
    chunk_count: int


@dataclass(frozen=True)
class IndexResult:
    """Outcome of indexing one file."""

    document_id: str
    chunk_count: int


@dataclass(frozen=True)
class AskResult:
    """Outcome of a question against the current index."""

    answer: GeneratedAnswer
    context: ContextPayload
    retrieval_count: int
    is_empty_index: bool
    is_empty_retrieval: bool


class DocumentAnalyzer:
    """Compose Retriever + ContextBuilder + AnswerGenerator for the UI."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: AnswerGenerator | None = None,
        top_k: int | None = None,
    ):
        self._retriever = retriever if retriever is not None else Retriever()
        self._generator = generator if generator is not None else AnswerGenerator()
        self._top_k = DEFAULT_TOP_K if top_k is None else top_k

    def save_upload(self, filename: str, data: bytes) -> Path:
        """Persist uploaded bytes under ``DOCUMENTS_DIR`` using the base filename."""
        name = Path(filename).name
        if not name or name in {".", ".."}:
            raise ValueError("A valid filename is required")
        destination = config.DOCUMENTS_DIR / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    def index_file(self, path, document_id: str | None = None) -> IndexResult:
        """Ingest a PDF/TXT file and replace any prior chunks for that document id.

        When ``document_id`` is omitted, the full filename is used so citations
        match the name shown in the document list.
        """
        path = Path(path)
        doc_id = path.name if document_id is None else document_id
        chunk_count = self._retriever.add_file(path, document_id=doc_id)
        return IndexResult(document_id=doc_id, chunk_count=chunk_count)

    def list_documents(self) -> list[IndexedDocument]:
        """Return the current index inventory (unique IDs and chunk counts)."""
        return [
            IndexedDocument(document_id=doc_id, chunk_count=count)
            for doc_id, count in self._retriever.list_documents()
        ]

    def chunk_count(self) -> int:
        """Return the total number of indexed chunks."""
        return self._retriever.count()

    def delete_document(self, document_id: str) -> int:
        """Remove every indexed chunk for ``document_id``."""
        return self._retriever.delete_document(document_id)

    def clear(self) -> int:
        """Remove every indexed chunk."""
        return self._retriever.clear()

    def ask(self, question: str, top_k: int | None = None) -> AskResult:
        """Retrieve, build context, and generate an answer for ``question``."""
        k = self._top_k if top_k is None else top_k
        is_empty_index = self._retriever.count() == 0
        hits = self._retriever.search(question, top_k=k)
        context = build_context(hits)
        answer = self._generator.generate(question, context)
        return AskResult(
            answer=answer,
            context=context,
            retrieval_count=len(hits),
            is_empty_index=is_empty_index,
            is_empty_retrieval=len(hits) == 0,
        )


def groq_api_key_configured() -> bool:
    """Return whether a Groq API key is set, without exposing the value."""
    key = config.GROQ_API_KEY
    return key is not None and key != ""
