"""Tests for app.storage.vector_store."""

import numpy as np
import pytest

from app.chunking.chunker import Chunk
from app.storage.vector_store import VectorStore, chunk_id

DIM = 8


def _chunk(document_id, chunk_index, text):
    return Chunk(document_id=document_id, chunk_index=chunk_index, text=text)


def _embedding(index, dim: int = DIM) -> list:
    """Deterministic unit vector pointing in a distinct direction."""
    vector = np.zeros(dim, dtype="float32")
    vector[index % dim] = 1.0
    return vector.tolist()


@pytest.fixture
def store(tmp_path):
    return VectorStore(path=tmp_path / "faiss", collection_name="test")


def _sample_chunks():
    return [
        _chunk("doc-a", 0, "Word zero alpha."),
        _chunk("doc-a", 1, "Word one beta."),
        _chunk("doc-b", 0, "Word two gamma."),
    ]


def test_chunk_id_is_deterministic():
    assert chunk_id("doc-1", 0) == "doc-1::0"
    assert chunk_id("doc-2", 5) == "doc-2::5"
    assert chunk_id("doc-1", 0) == chunk_id("doc-1", 0)
    assert chunk_id("doc-1", 0) != chunk_id("doc-1", 1)


def test_new_store_is_empty(store):
    assert store.count() == 0
    assert store.query([0.0] * DIM, top_k=5) == []


def test_upsert_chunks_increases_count(store):
    embeddings = [_embedding(0), _embedding(1), _embedding(2)]
    added = store.upsert_chunks(_sample_chunks(), embeddings)
    assert added == 3
    assert store.count() == 3


def test_upsert_returns_number_of_chunks(store):
    added = store.upsert_chunks(
        [_chunk("doc-1", 0, "text")], [_embedding(1)]
    )
    assert added == 1


def test_upsert_same_ids_does_not_duplicate(store):
    chunks = _sample_chunks()
    embeddings = [_embedding(0), _embedding(1), _embedding(2)]
    store.upsert_chunks(chunks, embeddings)
    store.upsert_chunks(chunks, embeddings)
    assert store.count() == 3


def test_upsert_replaces_existing_text(store):
    store.upsert_chunks([_chunk("doc-1", 0, "original")], [_embedding(1)])
    store.upsert_chunks([_chunk("doc-1", 0, "updated")], [_embedding(1)])
    assert store.count() == 1
    hits = store.query(_embedding(1), top_k=1)
    assert hits[0].text == "updated"


def test_upsert_skips_blank_chunks(store):
    added = store.upsert_chunks(
        [_chunk("doc-1", 0, "   \n  ")], [_embedding(0)]
    )
    assert added == 0
    assert store.count() == 0


def test_upsert_empty_input_is_noop(store):
    assert store.upsert_chunks([], []) == 0
    assert store.count() == 0


def test_upsert_embedding_count_mismatch_raises(store):
    with pytest.raises(ValueError, match="embeddings"):
        store.upsert_chunks(_sample_chunks(), [_embedding(0)])


def test_query_returns_metadadata_for_each_hit(store):
    store.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])
    hits = store.query(_embedding(1), top_k=3)
    assert len(hits) == 3
    for hit in hits:
        assert isinstance(hit.document_id, str)
        assert isinstance(hit.chunk_index, int)
        assert isinstance(hit.text, str)
        assert isinstance(hit.distance, float)
        assert isinstance(hit.score, float)
        assert hit.score == pytest.approx(1.0 - hit.distance)


def test_query_orders_by_distance(store):
    store.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])
    hits = store.query(_embedding(1), top_k=3)
    assert hits[0].text == "Word one beta."
    assert hits[0].document_id == "doc-a"
    assert hits[0].chunk_index == 1
    distances = [hit.distance for hit in hits]
    assert distances == sorted(distances)


def test_query_top_k_limits_results(store):
    chunks = [_chunk("doc-1", i, f"chunk {i}") for i in range(6)]
    store.upsert_chunks(chunks, [_embedding(i) for i in range(6)])
    assert len(store.query(_embedding(0), top_k=3)) == 3
    assert len(store.query(_embedding(0), top_k=1)) == 1


def test_query_top_k_larger_than_collection(store):
    store.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])
    assert len(store.query(_embedding(0), top_k=50)) == 3


def test_delete_document_removes_only_that_document(store):
    store.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])
    removed = store.delete_document("doc-a")
    assert removed == 2
    assert store.count() == 1
    hits = store.query(_embedding(0), top_k=10)
    assert all(hit.document_id == "doc-b" for hit in hits)


def test_delete_missing_document(store):
    assert store.delete_document("nope") == 0
    assert store.count() == 0


def test_clear_empties_collection(store):
    store.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])
    removed = store.clear()
    assert removed == 3
    assert store.count() == 0
    assert store.query(_embedding(0), top_k=5) == []


def test_clear_empty_collection(store):
    assert store.clear() == 0


def test_store_persists_across_instances(tmp_path):
    path = tmp_path / "faiss"
    first = VectorStore(path=path, collection_name="persisted")
    first.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])

    second = VectorStore(path=path, collection_name="persisted")
    assert second.count() == 3
    hits = second.query(_embedding(1), top_k=1)
    assert hits[0].text == "Word one beta."


def test_faiss_index_type_and_dimension(store):
    store.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])
    import faiss
    assert isinstance(store._index, faiss.IndexIDMap2)
    assert store._dim == DIM
    assert store._index.d == DIM


def test_faiss_dimension_mismatch_raises(store):
    store.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])
    with pytest.raises(ValueError, match="embedding dimension"):
        store.upsert_chunks([_chunk("doc-c", 0, "mismatched")], [[1.0] * 128])


def test_faiss_query_dimension_mismatch_raises(store):
    store.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])
    with pytest.raises(ValueError, match="query dimension"):
        store.query([1.0] * 128, top_k=1)


def test_faiss_idempotent_reindex_does_not_leak_ids(store):
    chunks = [_chunk("doc-1", 0, "initial")]
    store.upsert_chunks(chunks, [_embedding(0)])
    assert store.count() == 1
    # Repeatedly update the same chunk
    for i in range(5):
        store.upsert_chunks([_chunk("doc-1", 0, f"update {i}")], [_embedding(0)])
        assert store.count() == 1
    hits = store.query(_embedding(0), top_k=1)
    assert hits[0].text == "update 4"


def test_faiss_persistence_fidelity(tmp_path):
    path = tmp_path / "fidelity"
    first = VectorStore(path=path, collection_name="fidelity")
    first.upsert_chunks(_sample_chunks(), [_embedding(0), _embedding(1), _embedding(2)])
    first_hits = first.query(_embedding(1), top_k=3)

    second = VectorStore(path=path, collection_name="fidelity")
    second_hits = second.query(_embedding(1), top_k=3)

    assert len(first_hits) == len(second_hits)
    for h1, h2 in zip(first_hits, second_hits):
        assert h1.text == h2.text
        assert h1.document_id == h2.document_id
        assert h1.chunk_index == h2.chunk_index
        assert h1.score == pytest.approx(h2.score)
        assert h1.distance == pytest.approx(h2.distance)