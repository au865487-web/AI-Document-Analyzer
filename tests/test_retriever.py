"""Tests for app.retrieval.retriever.

These are integration tests: they index real text and search it with the
real ``all-MiniLM-L6-v2`` embedding model against a temporary local
FAISS directory.
"""

import os

import pytest

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import fitz

from app.embeddings.embedder import DEFAULT_MODEL, Embedder
from app.ingestion.extractor import ExtractionError
from app.retrieval.retriever import DEFAULT_TOP_K, Retriever
from app.storage.vector_store import SearchResult, VectorStore

HEART_TEXT = (
    "The heart is a muscular organ that pumps blood through the blood "
    "vessels of the circulatory system. It beats continuously to supply "
    "oxygen to every cell in the body."
)
FOREST_TEXT = (
    "Maple trees turn brilliant orange and red every autumn. Their leaves "
    "fall to the ground as the days grow shorter and the nights turn cold."
)
TECH_TEXT = (
    "Semantic search combines neural embeddings with a vector index to "
    "find documents that match the meaning of a query rather than its "
    "exact words."
)


@pytest.fixture(scope="module")
def embedder():
    return Embedder(model_name=DEFAULT_MODEL)


@pytest.fixture
def store(tmp_path):
    return VectorStore(path=tmp_path / "faiss", collection_name="retriever")


@pytest.fixture
def retriever(store, embedder):
    return Retriever(store=store, embedder=embedder)


def _long_text():
    return " ".join(
        f"The quick brown fox jumps over the lazy dog number {i}." for i in range(30)
    )


def test_default_top_k():
    assert DEFAULT_TOP_K == 10


def test_add_document_returns_chunk_count(retriever):
    added = retriever.add_document("long", _long_text())
    assert added > 1
    assert retriever.count() == added


def test_add_document_blank_text_indexes_nothing(retriever):
    assert retriever.add_document("empty", "   \n  ") == 0
    assert retriever.count() == 0


def test_search_empty_collection_returns_empty(retriever):
    assert retriever.search("anything at all") == []


def test_search_blank_query_returns_empty(retriever):
    retriever.add_document("heart", HEART_TEXT)
    assert retriever.search("   ") == []
    assert retriever.search("") == []
    assert retriever.search(123) == []


def test_invalid_top_k_raises(retriever):
    retriever.add_document("heart", HEART_TEXT)
    for bad in (0, -1, 2.5, "5", True):
        with pytest.raises(ValueError, match="top_k"):
            retriever.search("heart blood", top_k=bad)


def test_reindex_same_document_does_not_duplicate(retriever):
    first = retriever.add_document("long", _long_text())
    second = retriever.add_document("long", _long_text()[:400])
    assert retriever.count() == second
    assert second < first


def test_search_returns_top_k_results(retriever):
    retriever.add_document("heart", HEART_TEXT)
    retriever.add_document("forest", FOREST_TEXT)
    retriever.add_document("tech", TECH_TEXT)
    hits = retriever.search("blood pump circulation", top_k=2)
    assert len(hits) == 2
    shortest = retriever.search("blood pump circulation", top_k=1)
    assert len(shortest) == 1


def test_search_results_are_search_results(retriever):
    retriever.add_document("heart", HEART_TEXT)
    find = retriever.search("cardiovascular system")
    assert find
    for hit in find:
        assert isinstance(hit, SearchResult)


def test_search_returns_best_match_first(retriever):
    retriever.add_document("heart", HEART_TEXT)
    retriever.add_document("forest", FOREST_TEXT)
    retriever.add_document("tech", TECH_TEXT)
    hits = retriever.search("pumping blood through arteries")
    assert hits[0].document_id == "heart"
    distances = [hit.distance for hit in hits]
    assert distances == sorted(distances)


def test_search_results_include_metadata_and_score(retriever):
    retriever.add_document("heart", HEART_TEXT)
    hits = retriever.search("heart muscle", top_k=3)
    assert hits
    for hit in hits:
        assert isinstance(hit.document_id, str)
        assert isinstance(hit.chunk_index, int)
        assert isinstance(hit.text, str) and hit.text.strip()
        assert isinstance(hit.distance, float)
        assert isinstance(hit.score, float)
        assert hit.score == pytest.approx(1.0 - hit.distance)


def test_semantic_retrieval_across_documents(retriever):
    retriever.add_document("heart", HEART_TEXT)
    retriever.add_document("forest", FOREST_TEXT)
    retriever.add_document("tech", TECH_TEXT)

    heart_hit = retriever.search("how does the heart pump blood?", top_k=1)
    assert heart_hit[0].document_id == "heart"

    forest_hit = retriever.search("autumn colors of maple leaves", top_k=1)
    assert forest_hit[0].document_id == "forest"

    tech_hit = retriever.search("embedding based semantic search", top_k=1)
    assert tech_hit[0].document_id == "tech"


def test_retriever_delete_document(retriever):
    retriever.add_document("heart", HEART_TEXT)
    retriever.add_document("forest", FOREST_TEXT)
    removed = retriever.delete_document("heart")
    assert removed == retriever.count()
    hits = retriever.search("heart blood circulation", top_k=5)
    assert all(hit.document_id == "forest" for hit in hits)


def test_retriever_clear(retriever):
    retriever.add_document("heart", HEART_TEXT)
    retriever.add_document("forest", FOREST_TEXT)
    removed = retriever.clear()
    assert removed > 0
    assert retriever.count() == 0
    assert retriever.search("anything") == []


def test_add_file_pdf_end_to_end(retriever, tmp_path):
    path = tmp_path / "science.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "The cardiovascular system uses arteries and veins to transport oxygenated blood throughout the body.")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Photosynthesis allows green plants and trees to convert sunlight and carbon dioxide into chemical energy.")
    doc.save(str(path))
    doc.close()

    added = retriever.add_file(path)
    assert added >= 2
    assert retriever.count() == added

    # Query matching page 1
    heart_hits = retriever.search("arteries oxygenated blood vessels", top_k=1)
    assert heart_hits
    assert heart_hits[0].document_id == "science"
    assert heart_hits[0].page_number == 1

    # Query matching page 2
    plant_hits = retriever.search("photosynthesis sunlight plants carbon dioxide", top_k=1)
    assert plant_hits
    assert plant_hits[0].document_id == "science"
    assert plant_hits[0].page_number == 2


def test_add_file_txt_end_to_end(retriever, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(
        "Neural networks use backpropagation and gradient descent to update their internal weights during training.",
        encoding="utf-8",
    )

    added = retriever.add_file(path)
    assert added >= 1

    hits = retriever.search("backpropagation gradient descent training", top_k=1)
    assert hits
    assert hits[0].document_id == "notes"
    assert hits[0].page_number is None


def test_add_file_custom_document_id(retriever, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Unique text content about distributed systems.", encoding="utf-8")

    added = retriever.add_file(path, document_id="custom-dist-sys")
    assert added >= 1

    hits = retriever.search("distributed systems", top_k=1)
    assert hits
    assert hits[0].document_id == "custom-dist-sys"


def test_add_file_idempotent_reindex(retriever, tmp_path):
    path = tmp_path / "manual.txt"
    path.write_text("Operating instructions for industrial machinery.", encoding="utf-8")

    first = retriever.add_file(path)
    second = retriever.add_file(path)
    assert first == second
    assert retriever.count() == first


def test_add_file_preserves_old_on_extraction_error(retriever, tmp_path):
    valid_path = tmp_path / "valid.txt"
    valid_path.write_text("Important preserved document text.", encoding="utf-8")
    retriever.add_file(valid_path, document_id="doc-keep")
    assert retriever.count() >= 1

    # Attempt replacement with non-existent file
    with pytest.raises(ExtractionError):
        retriever.add_file(tmp_path / "nonexistent.txt", document_id="doc-keep")

    # Verify old document is still completely present in the index
    assert retriever.count() >= 1
    hits = retriever.search("preserved document text", top_k=1)
    assert hits
    assert hits[0].document_id == "doc-keep"


def test_add_file_missing_file_raises(retriever, tmp_path):
    with pytest.raises(ExtractionError, match="not found"):
        retriever.add_file(tmp_path / "missing.pdf")
