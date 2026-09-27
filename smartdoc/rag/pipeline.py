"""End-to-end RAG pipeline: ingest -> embed -> index -> retrieve -> generate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from smartdoc.config import Settings
from smartdoc.document_processing.chunker import chunk_documents
from smartdoc.document_processing.extractor import extract_documents
from smartdoc.embeddings.encoder import EmbeddingEncoder
from smartdoc.exceptions import NoRelevantContextError, SmartDocError
from smartdoc.llm.gemini_client import GeminiClient
from smartdoc.llm.prompts import NOT_FOUND_MESSAGE, build_qa_prompt, format_context
from smartdoc.logging_setup import get_logger
from smartdoc.models import DocumentRecord, RAGAnswer, RetrievedChunk, TextChunk
from smartdoc.retrieval.retriever import SemanticRetriever
from smartdoc.vectorstore.faiss_store import FaissVectorStore

logger = get_logger()


@dataclass
class IngestedCorpus:
    fingerprint: str
    documents: list[DocumentRecord]
    chunks: list[TextChunk]
    store: FaissVectorStore
    encoder: EmbeddingEncoder
    extra: dict = field(default_factory=dict)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def document_names(self) -> list[str]:
        return [d.document_name for d in self.documents]


def fingerprint_files(files: list) -> str:
    digest = hashlib.sha256()
    for uploaded in files:
        name = getattr(uploaded, "name", "")
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(str(len(data)).encode("ascii"))
        digest.update(data)
    return digest.hexdigest()


class RAGPipeline:
    def __init__(self, settings: Settings, encoder: EmbeddingEncoder | None = None) -> None:
        self.settings = settings
        self.encoder = encoder or EmbeddingEncoder(settings.embedding_model)
        self.llm = GeminiClient(settings.gemini_api_key, settings.gemini_model)

    def ingest(self, files: list, *, existing: IngestedCorpus | None = None) -> IngestedCorpus:
        fp = fingerprint_files(files)
        if existing is not None and existing.fingerprint == fp and existing.store.is_ready:
            logger.info("Reusing existing FAISS index for unchanged documents")
            return existing

        documents = extract_documents(
            files, max_file_size_bytes=self.settings.max_file_size_bytes
        )
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        embeddings = self.encoder.encode([c.text for c in chunks])
        store = FaissVectorStore()
        store.build(embeddings, chunks)
        return IngestedCorpus(
            fingerprint=fp,
            documents=documents,
            chunks=chunks,
            store=store,
            encoder=self.encoder,
        )

    def ask(self, corpus: IngestedCorpus, question: str) -> RAGAnswer:
        retriever = SemanticRetriever(
            corpus.store,
            corpus.encoder,
            top_k=self.settings.top_k,
            min_similarity=self.settings.min_similarity,
        )
        retrieved = retriever.retrieve(question)
        if not retrieved:
            logger.info("No relevant context; skipping Gemini call")
            return RAGAnswer(
                question=question,
                answer=NOT_FOUND_MESSAGE,
                sources=[],
                found_in_documents=False,
                retrieval_skipped_llm=True,
            )

        context = format_context(retrieved, self.settings.max_context_chars)
        prompt = build_qa_prompt(question, context)
        try:
            raw = self.llm.generate(prompt)
        except SmartDocError:
            raise

        found = not _is_not_found(raw)
        return RAGAnswer(
            question=question,
            answer=raw,
            sources=retrieved,
            found_in_documents=found,
            retrieval_skipped_llm=False,
            extra={"context_chars": len(context)},
        )

    def retrieve_only(self, corpus: IngestedCorpus, question: str) -> list[RetrievedChunk]:
        retriever = SemanticRetriever(
            corpus.store,
            corpus.encoder,
            top_k=self.settings.top_k,
            min_similarity=self.settings.min_similarity,
        )
        hits = retriever.retrieve(question)
        if not hits:
            raise NoRelevantContextError(
                "No relevant chunks",
                user_message=NOT_FOUND_MESSAGE,
            )
        return hits


def _is_not_found(text: str) -> bool:
    lowered = text.strip().lower()
    return NOT_FOUND_MESSAGE.lower() in lowered or lowered.startswith(
        "the requested information was not found"
    )
