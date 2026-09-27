"""Semantic similarity retrieval over the FAISS index."""

from __future__ import annotations

from smartdoc.embeddings.encoder import EmbeddingEncoder
from smartdoc.exceptions import InvalidQuestionError, RetrievalError
from smartdoc.logging_setup import get_logger
from smartdoc.models import RetrievedChunk
from smartdoc.vectorstore.faiss_store import FaissVectorStore

logger = get_logger()


class SemanticRetriever:
    def __init__(
        self,
        store: FaissVectorStore,
        encoder: EmbeddingEncoder,
        *,
        top_k: int,
        min_similarity: float,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        cleaned = (question or "").strip()
        if not cleaned:
            raise InvalidQuestionError(
                "Empty question",
                user_message="Please enter a question about the uploaded documents.",
            )
        if len(cleaned) < 3:
            raise InvalidQuestionError(
                "Question too short",
                user_message="Please enter a more specific question.",
            )
        if not self.store.is_ready:
            raise RetrievalError(
                "No index",
                user_message="No documents are indexed yet. Upload and process documents first.",
            )

        query_vec = self.encoder.encode([cleaned])
        hits = self.store.search(query_vec, self.top_k)
        retrieved: list[RetrievedChunk] = []
        rank = 1
        for faiss_id, score in hits:
            if score < self.min_similarity:
                continue
            chunk = self.store.get_chunk(faiss_id)
            retrieved.append(RetrievedChunk(chunk=chunk, score=score, rank=rank))
            rank += 1

        logger.info(
            "Retrieved %s/%s chunks above threshold %.2f for question length %s",
            len(retrieved),
            len(hits),
            self.min_similarity,
            len(cleaned),
        )
        return retrieved
