"""Tests for app.embeddings.embedder.

The integration tests load the real ``all-MiniLM-L6-v2`` model (downloaded
on first run) so embedding shape, dtype, and query behavior are verified
against the actual model.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from app import config
from app.embeddings.embedder import DEFAULT_MODEL, Embedder


def test_default_model_name():
    assert DEFAULT_MODEL == "all-MiniLM-L6-v2"


def test_model_name_resolution_env(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "some-embedding-model")
    assert Embedder().model_name == "some-embedding-model"


def test_model_name_resolution_default(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "")
    assert Embedder().model_name == DEFAULT_MODEL


def test_explicit_model_name_overrides_config(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "from-config")
    assert Embedder(model_name="explicit").model_name == "explicit"


@pytest.fixture(scope="module")
def embedder():
    return Embedder(model_name=DEFAULT_MODEL)


def test_embed_documents_shape_and_type(embedder):
    vectors = embedder.embed_documents(["The quick brown fox jumps.", "A lazy dog."])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 384)
    assert np.issubdtype(vectors.dtype, np.floating)
    assert np.isfinite(vectors).all()


def test_embed_documents_single_item(embedder):
    vectors = embedder.embed_documents(["Just one text."])
    assert vectors.shape == (1, 384)


def test_embed_query_shape_and_type(embedder):
    vector = embedder.embed_query("What is a document?")
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (384,)
    assert np.issubdtype(vector.dtype, np.floating)
    assert np.isfinite(vector).all()


def test_embed_query_matches_document_vector(embedder):
    text = "Semantic search over documents."
    query_vector = embedder.embed_query(text)
    doc_vector = embedder.embed_documents([text])[0]
    assert np.allclose(query_vector, doc_vector, atol=1e-5)


def test_embed_documents_is_deterministic(embedder):
    texts = ["First paragraph.", "Second paragraph."]
    first = embedder.embed_documents(texts)
    second = embedder.embed_documents(texts)
    assert np.array_equal(first, second)


def test_embed_query_is_deterministic(embedder):
    first = embedder.embed_query("repeat me")
    second = embedder.embed_query("repeat me")
    assert np.array_equal(first, second)