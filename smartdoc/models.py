"""Core dataclasses used across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedPage:
    """One extracted unit of a document (PDF page, DOCX section, or TXT slice)."""

    document_name: str
    file_type: str
    text: str
    page_number: int | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class DocumentRecord:
    document_name: str
    file_type: str
    size_bytes: int
    page_count: int
    char_count: int
    pages: list[ExtractedPage] = field(default_factory=list)


@dataclass
class TextChunk:
    chunk_id: str
    document_name: str
    file_type: str
    text: str
    chunk_index: int
    page_number: int | None = None
    page_end: int | None = None
    start_line: int | None = None
    end_line: int | None = None

    def page_label(self) -> str:
        if self.page_number is None:
            return "N/A"
        if self.page_end is not None and self.page_end != self.page_number:
            return f"{self.page_number}-{self.page_end}"
        return str(self.page_number)

    def source_label(self) -> str:
        return f"{self.document_name} (page {self.page_label()})"


@dataclass
class RetrievedChunk:
    chunk: TextChunk
    score: float
    rank: int


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    found_in_documents: bool
    retrieval_skipped_llm: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentInsights:
    summary: str
    key_points: list[str]
    structured_notes: str = ""
    used_map_reduce: bool = False
    document_names: list[str] = field(default_factory=list)
