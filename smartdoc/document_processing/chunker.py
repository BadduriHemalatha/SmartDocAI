"""Intelligent overlapping chunking that preserves source metadata."""

from __future__ import annotations

import re
import uuid

from smartdoc.document_processing.preprocessor import preprocess_text
from smartdoc.exceptions import EmptyDocumentError
from smartdoc.logging_setup import get_logger
from smartdoc.models import DocumentRecord, ExtractedPage, TextChunk

logger = get_logger()

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_documents(
    documents: list[DocumentRecord],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    all_chunks: list[TextChunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    if not all_chunks:
        raise EmptyDocumentError(
            "No chunks produced",
            user_message="The uploaded documents did not contain enough text to index.",
        )
    logger.info("Created %s chunks from %s documents", len(all_chunks), len(documents))
    return all_chunks


def chunk_document(
    document: DocumentRecord,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    pieces: list[tuple[str, ExtractedPage]] = []
    for page in document.pages:
        text = preprocess_text(page.text)
        if text:
            pieces.append((text, page))

    if not pieces:
        return []

    windows = _windows_from_pages(pieces, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[TextChunk] = []
    for index, window in enumerate(windows):
        chunks.append(
            TextChunk(
                chunk_id=str(uuid.uuid4()),
                document_name=document.document_name,
                file_type=document.file_type,
                text=window["text"],
                chunk_index=index,
                page_number=window["page_number"],
                page_end=window["page_end"],
                start_line=window.get("start_line"),
                end_line=window.get("end_line"),
            )
        )
    return chunks


def _windows_from_pages(
    pieces: list[tuple[str, ExtractedPage]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Build overlapping windows while tracking originating page numbers."""
    units: list[dict] = []
    for text, page in pieces:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        for para in paragraphs:
            units.append(
                {
                    "text": para,
                    "page_number": page.page_number,
                    "start_line": page.start_line,
                    "end_line": page.end_line,
                }
            )

    windows: list[dict] = []
    buffer = ""
    pages: list[int | None] = []
    start_line = None
    end_line = None

    def flush(force: bool = False) -> None:
        nonlocal buffer, pages, start_line, end_line
        text = buffer.strip()
        if not text:
            return
        if not force and len(text) < chunk_size:
            return
        numbered = [p for p in pages if p is not None]
        windows.append(
            {
                "text": text,
                "page_number": numbered[0] if numbered else (pages[0] if pages else None),
                "page_end": numbered[-1] if numbered else None,
                "start_line": start_line,
                "end_line": end_line,
            }
        )
        if chunk_overlap > 0 and len(text) > chunk_overlap:
            overlap_text = text[-chunk_overlap:]
            buffer = overlap_text
            pages = pages[-1:] if pages else []
        else:
            buffer = ""
            pages = []
            start_line = None
            end_line = None

    for unit in units:
        candidate = (buffer + "\n\n" + unit["text"]).strip() if buffer else unit["text"]
        if len(candidate) <= chunk_size:
            buffer = candidate
            pages.append(unit["page_number"])
            if start_line is None:
                start_line = unit["start_line"]
            end_line = unit["end_line"]
            continue

        # Split oversized paragraphs into sentence/character windows.
        remaining = unit["text"]
        while remaining:
            space_left = chunk_size - (len(buffer) + (2 if buffer else 0))
            if space_left <= 40:
                flush(force=True)
                space_left = chunk_size
            piece, remaining = _take_split(remaining, max(space_left, 40))
            buffer = (buffer + "\n\n" + piece).strip() if buffer else piece
            pages.append(unit["page_number"])
            if start_line is None:
                start_line = unit["start_line"]
            end_line = unit["end_line"]
            if len(buffer) >= chunk_size:
                flush(force=True)

    if buffer.strip():
        flush(force=True)
    return windows


def _take_split(text: str, limit: int) -> tuple[str, str]:
    if len(text) <= limit:
        return text, ""
    window = text[:limit]
    sentences = _SENTENCE_SPLIT.split(window)
    if len(sentences) > 1:
        used = " ".join(sentences[:-1]).strip()
        if used:
            return used, text[len(used) :].lstrip()
    last_space = window.rfind(" ")
    if last_space > limit * 0.4:
        return text[:last_space].strip(), text[last_space:].lstrip()
    return window, text[limit:]
