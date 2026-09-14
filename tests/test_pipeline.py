"""Tests for app.pipeline.DocumentAnalyzer and VectorStore.list_documents.

Indexing tests use a temporary FAISS directory and the real embedder.
Ask tests inject a protocol-compatible Groq stub so no network calls occur.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from app.chunking.chunker import Chunk
from app.embeddings.embedder import DEFAULT_MODEL, Embedder
from app.generation.exceptions import ConfigurationError
from app.generation.generator import AnswerGenerator
from app.ingestion.extractor import ExtractionError
from app.pipeline import DocumentAnalyzer, groq_api_key_configured
from app.retrieval.retriever import Retriever
from app.storage.vector_store import VectorStore

DIM = 8


class _FakeCompletions:
    def __init__(self, response=None):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, content: str):
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice], usage=None)
        self.completions = _FakeCompletions(response=response)
        self.chat = SimpleNamespace(completions=self.completions)


def _chunk(document_id, chunk_index, text):
    return Chunk(document_id=document_id, chunk_index=chunk_index, text=text)


def _embedding(index, dim: int = DIM) -> list:
    vector = np.zeros(dim, dtype="float32")
    vector[index % dim] = 1.0
    return vector.tolist()


@pytest.fixture
def store(tmp_path):
    return VectorStore(path=tmp_path / "faiss", collection_name="pipeline")


@pytest.fixture(scope="module")
def embedder():
    return Embedder(model_name=DEFAULT_MODEL)


@pytest.fixture
def analyzer(store, embedder):
    retriever = Retriever(store=store, embedder=embedder)
    client = _FakeClient("The margin rose. [Source 1]")
    generator = AnswerGenerator(client=client)
    analyzer = DocumentAnalyzer(retriever=retriever, generator=generator)
    analyzer._fake_client = client  # test inspection only
    return analyzer


def test_list_documents_empty_store(store):
    assert store.list_documents() == []


def test_list_documents_unique_ids_and_chunk_counts(store):
    chunks = [
        _chunk("report.pdf", 0, "Alpha."),
        _chunk("report.pdf", 1, "Beta."),
        _chunk("notes.txt", 0, "Gamma."),
    ]
    store.upsert_chunks(chunks, [_embedding(0), _embedding(1), _embedding(2)])
    assert store.list_documents() == [("notes.txt", 1), ("report.pdf", 2)]


def test_list_documents_updates_after_delete(store):
    chunks = [
        _chunk("keep.pdf", 0, "Keep me."),
        _chunk("drop.pdf", 0, "Drop me."),
    ]
    store.upsert_chunks(chunks, [_embedding(0), _embedding(1)])
    store.delete_document("drop.pdf")
    assert store.list_documents() == [("keep.pdf", 1)]


def test_retriever_list_documents_delegates(store):
    retriever = Retriever(store=store)
    store.upsert_chunks(
        [_chunk("only.txt", 0, "Hello there.")],
        [_embedding(0)],
    )
    assert retriever.list_documents() == [("only.txt", 1)]


def test_index_file_and_list_documents(analyzer, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(
        "Neural networks use backpropagation to update weights during training.",
        encoding="utf-8",
    )
    result = analyzer.index_file(path)
    assert result.document_id == "notes.txt"
    assert result.chunk_count >= 1
    listed = analyzer.list_documents()
    assert listed[0].document_id == "notes.txt"
    assert listed[0].chunk_count == result.chunk_count


def test_index_file_reindex_replaces_chunks(analyzer, tmp_path):
    path = tmp_path / "manual.txt"
    path.write_text("Operating instructions for industrial machinery.", encoding="utf-8")
    first = analyzer.index_file(path)
    second = analyzer.index_file(path)
    assert first.chunk_count == second.chunk_count
    assert analyzer.chunk_count() == first.chunk_count
    docs = analyzer.list_documents()
    assert len(docs) == 1
    assert docs[0].chunk_count == first.chunk_count


def test_index_file_extraction_error(analyzer, tmp_path):
    with pytest.raises(ExtractionError):
        analyzer.index_file(tmp_path / "missing.pdf")
    assert analyzer.list_documents() == []


def test_ask_with_mock_generator(analyzer, tmp_path):
    path = tmp_path / "report.pdf.txt"
    # Use a .txt with a pdf-like name for filename-as-id without PDF bytes.
    path = tmp_path / "report.txt"
    path.write_text(
        "Operating profit margin rose by 4.2 percent this quarter.",
        encoding="utf-8",
    )
    analyzer.index_file(path)
    result = analyzer.ask("What happened to the margin?")
    assert result.is_empty_index is False
    assert result.is_empty_retrieval is False
    assert result.retrieval_count >= 1
    assert not result.context.is_empty
    assert "margin rose" in result.answer.answer.lower()
    assert result.answer.citations
    assert result.answer.citations[0].document_id == "report.txt"
    assert analyzer._fake_client.completions.calls


def test_ask_empty_index_does_not_call_groq(store, embedder):
    client = _FakeClient("should not be used")
    analyzer = DocumentAnalyzer(
        retriever=Retriever(store=store, embedder=embedder),
        generator=AnswerGenerator(client=client),
    )
    result = analyzer.ask("What happened?")
    assert result.is_empty_index is True
    assert result.is_empty_retrieval is True
    assert result.answer.is_empty_context is True
    assert client.completions.calls == []


class _IndexedEmptySearchRetriever:
    """Indexed chunks exist, but search returns no hits."""

    def count(self) -> int:
        return 2

    def search(self, query: str, top_k: int = 5):
        return []


def test_ask_empty_retrieval_does_not_call_groq():
    client = _FakeClient("should not be used")
    analyzer = DocumentAnalyzer(
        retriever=_IndexedEmptySearchRetriever(),
        generator=AnswerGenerator(client=client),
    )
    result = analyzer.ask("What happened?")
    assert result.is_empty_index is False
    assert result.is_empty_retrieval is True
    assert result.retrieval_count == 0
    assert result.context.is_empty is True
    assert result.answer.is_empty_context is True
    assert client.completions.calls == []


def test_ask_passes_explicit_top_k():
    recorded = {}

    class _RecordingRetriever:
        def count(self) -> int:
            return 0

        def search(self, query: str, top_k: int = 5):
            recorded["top_k"] = top_k
            return []

    client = _FakeClient("unused")
    analyzer = DocumentAnalyzer(
        retriever=_RecordingRetriever(),
        generator=AnswerGenerator(client=client),
        top_k=5,
    )
    analyzer.ask("What happened?", top_k=3)
    assert recorded["top_k"] == 3


def test_ask_blank_question_raises(analyzer):
    with pytest.raises(ValueError, match="question"):
        analyzer.ask("   ")


def test_save_upload_writes_basename(analyzer, tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "DOCUMENTS_DIR", tmp_path / "documents")
    saved = analyzer.save_upload(r"..\\evil\\notes.txt", b"hello")
    assert saved.name == "notes.txt"
    assert saved.read_bytes() == b"hello"
    assert saved.parent == tmp_path / "documents"


def test_groq_api_key_configured_does_not_return_key(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "GROQ_API_KEY", "super-secret-key")
    assert groq_api_key_configured() is True
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    assert groq_api_key_configured() is False


def test_missing_api_key_without_client_raises_configuration_error(
    store, embedder, tmp_path, monkeypatch
):
    from app import config

    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    path = tmp_path / "doc.txt"
    path.write_text("Indexed content about solar panels.", encoding="utf-8")
    analyzer = DocumentAnalyzer(
        retriever=Retriever(store=store, embedder=embedder),
        generator=AnswerGenerator(),
    )
    analyzer.index_file(path)
    with pytest.raises(ConfigurationError):
        analyzer.ask("What is indexed?")
