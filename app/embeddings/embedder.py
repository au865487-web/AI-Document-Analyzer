"""Thin embedding wrapper around sentence-transformers.

Provides a small, predictable interface for embedding document text and
queries with the same underlying model. The model name is configurable
through ``EMBEDDING_MODEL`` (environment or project configuration) and
defaults to ``all-MiniLM-L6-v2``, a small general-purpose model.

Vectors are returned as float32 NumPy arrays, L2-normalized so cosine
similarity equals a simple dot product downstream. The model is loaded
lazily on the first embed call; no caching or storage is implemented yet.
"""
from sentence_transformers import SentenceTransformer

from app import config

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Embed documents and queries using a sentence-transformer model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL or DEFAULT_MODEL
        self._model = None

    def embed_documents(self, texts):
        """Embed an iterable of document chunks.

        Returns an ``(N, dim)`` float32 NumPy array with one row per input.
        """
        model = self._load_model()
        return model.encode(list(texts), normalize_embeddings=True)

    def embed_query(self, text: str):
        """Embed a single query.

        Returns a ``(dim,)`` float32 NumPy array.
        """
        model = self._load_model()
        return model.encode([text], normalize_embeddings=True)[0]

    def _load_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model