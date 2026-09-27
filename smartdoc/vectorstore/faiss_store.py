"""FAISS vector index with metadata mapping."""

from __future__ import annotations

import numpy as np

from smartdoc.exceptions import VectorStoreError
from smartdoc.logging_setup import get_logger
from smartdoc.models import TextChunk

logger = get_logger()


class FaissVectorStore:
    def __init__(self) -> None:
        self._index = None
        self._chunks: list[TextChunk] = []
        self._dimension: int | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self.size > 0

    @property
    def chunks(self) -> list[TextChunk]:
        return list(self._chunks)

    def build(self, embeddings: np.ndarray, chunks: list[TextChunk]) -> None:
        if embeddings is None or len(embeddings) == 0:
            raise VectorStoreError(
                "No embeddings to index",
                user_message="Could not build the search index because no embeddings were produced.",
            )
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "Embedding/chunk count mismatch",
                user_message="Internal error: embeddings and document chunks are out of sync.",
            )
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError(
                "faiss is not installed",
                user_message="FAISS is not installed. Install dependencies from requirements.txt.",
            ) from exc

        try:
            vectors = np.ascontiguousarray(embeddings.astype("float32"))
            # Cosine similarity via inner product on L2-normalized vectors.
            faiss.normalize_L2(vectors)
            dimension = vectors.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(vectors)
        except Exception as exc:
            logger.exception("FAISS index build failed")
            raise VectorStoreError(
                f"FAISS indexing failed: {exc}",
                user_message="Failed to build the FAISS vector index.",
            ) from exc

        self._index = index
        self._chunks = list(chunks)
        self._dimension = dimension
        logger.info("FAISS index ready: %s vectors, dim=%s", self.size, dimension)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if not self.is_ready:
            raise VectorStoreError(
                "Index not ready",
                user_message="Please process documents before asking questions.",
            )
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("faiss is not installed") from exc

        try:
            vector = np.ascontiguousarray(query_embedding.astype("float32"))
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)
            k = max(1, min(top_k, self.size))
            scores, indices = self._index.search(vector, k)
        except Exception as exc:
            logger.exception("FAISS search failed")
            raise VectorStoreError(
                f"FAISS search failed: {exc}",
                user_message="Semantic search failed. Try processing the documents again.",
            ) from exc

        results: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if int(idx) < 0:
                continue
            results.append((int(idx), float(score)))
        return results

    def get_chunk(self, index: int) -> TextChunk:
        try:
            return self._chunks[index]
        except IndexError as exc:
            raise VectorStoreError("Invalid FAISS id") from exc
