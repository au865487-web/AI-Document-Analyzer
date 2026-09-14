"""FAISS-backed vector storage for document chunks.

Wraps a local persistent FAISS index (by default under
``DATA_DIR / "faiss"``) paired with a JSON-based metadata store so the rest
of the application never touches FAISS APIs directly. Chunks are stored with
deterministic IDs built from their document id and chunk index, which makes
re-indexing a document idempotent: upserting the same chunk again updates it
in place instead of duplicating it.

Queries return cosine distances (``0`` means identical; lower is better)
plus a convenience ``score`` of ``1 - distance`` (cosine similarity with
normalized embeddings; higher is better).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app import config
from app.chunking.chunker import Chunk

COLLECTION_NAME = "document_chunks"
DISTANCE_METRIC = "inner_product"


def chunk_id(document_id: str, chunk_index: int) -> str:
    """Return the deterministic, stable ID for ``chunk_index`` of a document."""
    return f"{document_id}::{chunk_index}"


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and how closely it matched the query."""

    text: str
    document_id: str
    chunk_index: int
    distance: float
    score: float
    page_number: int | None = None


class VectorStore:
    """Persistent vector store backed by FAISS IndexIDMap2 and metadata sidecar."""

    def __init__(
        self,
        path=None,
        collection_name: str = COLLECTION_NAME,
        dim: int | None = None,
    ):
        self._path = Path(path or config.FAISS_DIR)
        self._collection_name = collection_name
        self._dim = dim
        self._index: faiss.IndexIDMap2 | None = None
        self._metadata: dict[str, dict] = {}
        self._chunk_to_id: dict[str, int] = {}
        self._next_id: int = 0

        self._index_file = self._path / f"{self._collection_name}.faiss"
        self._meta_file = self._path / f"{self._collection_name}.json"

        self._load_if_exists()
        if self._index is None and self._dim is not None:
            self._init_index(self._dim)

    def _init_index(self, dim: int):
        flat_index = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIDMap2(flat_index)
        self._dim = dim

    def _load_if_exists(self):
        if self._index_file.is_file() and self._meta_file.is_file():
            try:
                loaded_index = faiss.read_index(str(self._index_file))
                if not isinstance(loaded_index, faiss.IndexIDMap2):
                    loaded_index = faiss.IndexIDMap2(loaded_index)
                self._index = loaded_index
                self._dim = self._index.d

                raw_meta = json.loads(self._meta_file.read_text(encoding="utf-8"))
                self._metadata = raw_meta.get("metadata", {})
                self._chunk_to_id = raw_meta.get("chunk_to_id", {})
                self._next_id = int(raw_meta.get("next_id", 0))
                if self._dim is None and "dim" in raw_meta:
                    self._dim = int(raw_meta["dim"])
            except Exception:
                self._index = None

    def _save(self):
        self._path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_file))
        payload = {
            "dim": self._dim,
            "next_id": self._next_id,
            "metadata": self._metadata,
            "chunk_to_id": self._chunk_to_id,
        }
        tmp_meta = self._path / f"{self._collection_name}.json.tmp"
        tmp_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_meta.replace(self._meta_file)

    def count(self) -> int:
        """Return the number of chunks stored in the collection."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def upsert_chunks(self, chunks: list[Chunk], embeddings) -> int:
        """Store ``chunks`` with their ``embeddings``, returning the count.

        Chunks keep their deterministic IDs, so storing an already-stored
        chunk replaces it rather than creating a duplicate. Empty chunks
        (including whitespace-only text) are skipped.
        """
        if not chunks:
            return 0

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"expected {len(chunks)} embeddings for {len(chunks)} chunks, "
                f"got {vectors.shape[0]}"
            )

        valid_indices = [i for i, c in enumerate(chunks) if c.text.strip()]
        if not valid_indices:
            return 0

        valid_chunks = [chunks[i] for i in valid_indices]
        valid_vectors = vectors[valid_indices]

        dim = valid_vectors.shape[1]
        if self._index is None:
            self._init_index(dim)
        elif dim != self._dim:
            raise ValueError(
                f"expected embedding dimension {self._dim}, got {dim}"
            )

        # Handle idempotency: if chunk_id already stored, remove old vector
        for c in valid_chunks:
            cid = chunk_id(c.document_id, c.chunk_index)
            if cid in self._chunk_to_id:
                old_int_id = self._chunk_to_id[cid]
                self._index.remove_ids(np.array([old_int_id], dtype=np.int64))
                self._metadata.pop(str(old_int_id), None)
                self._chunk_to_id.pop(cid, None)

        new_ids = []
        for c in valid_chunks:
            int_id = self._next_id
            self._next_id += 1
            cid = chunk_id(c.document_id, c.chunk_index)
            self._chunk_to_id[cid] = int_id
            self._metadata[str(int_id)] = {
                "text": c.text,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "chunk_id": cid,
                "page_number": c.page_number,
            }
            new_ids.append(int_id)

        ids_array = np.array(new_ids, dtype=np.int64)
        self._index.add_with_ids(valid_vectors, ids_array)
        self._save()
        return len(valid_chunks)

    def delete_document(self, document_id: str) -> int:
        """Delete every chunk belonging to ``document_id``; return how many."""
        to_delete_int_ids = []
        to_delete_cids = []

        for int_id_str, meta in list(self._metadata.items()):
            if meta.get("document_id") == document_id:
                to_delete_int_ids.append(int(int_id_str))
                to_delete_cids.append(meta.get("chunk_id"))

        if not to_delete_int_ids:
            return 0

        if self._index is not None:
            self._index.remove_ids(np.array(to_delete_int_ids, dtype=np.int64))

        for int_id in to_delete_int_ids:
            self._metadata.pop(str(int_id), None)
        for cid in to_delete_cids:
            if cid:
                self._chunk_to_id.pop(cid, None)

        self._save()
        return len(to_delete_int_ids)

    def clear(self) -> int:
        """Delete every chunk in the collection; return how many were removed."""
        total = self.count()
        if total == 0:
            return 0

        if self._index is not None:
            self._index.reset()

        self._metadata.clear()
        self._chunk_to_id.clear()
        self._save()
        return total

    def query(self, query_embedding, top_k: int) -> list[SearchResult]:
        """Return the ``top_k`` chunks closest to ``query_embedding``.

        An empty collection yields an empty list. Results are ordered by
        distance (closest first).
        """
        if self.count() == 0:
            return []

        top_k = min(top_k, self.count())
        if top_k < 1:
            return []

        query_vector = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if query_vector.shape[1] != self._dim:
            raise ValueError(
                f"expected query dimension {self._dim}, got {query_vector.shape[1]}"
            )

        sims, indices = self._index.search(query_vector, top_k)

        hits = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata.get(str(idx))
            if not meta:
                continue
            score = float(sim)
            distance = float(1.0 - score)
            hits.append(
                SearchResult(
                    text=meta["text"],
                    document_id=meta["document_id"],
                    chunk_index=int(meta["chunk_index"]),
                    distance=distance,
                    score=score,
                    page_number=meta.get("page_number"),
                )
            )

        hits.sort(key=lambda hit: hit.distance)
        return hits

    def list_documents(self) -> list[tuple[str, int]]:
        """Return unique document IDs with their current indexed chunk counts.

        Inventory is derived from the existing metadata sidecar (the same
        in-memory state used by query/delete). There is no separate registry.
        Results are sorted by document_id for stable display.
        """
        counts: dict[str, int] = {}
        for meta in self._metadata.values():
            document_id = meta.get("document_id")
            if not document_id:
                continue
            counts[document_id] = counts.get(document_id, 0) + 1
        return sorted(counts.items(), key=lambda item: item[0])
