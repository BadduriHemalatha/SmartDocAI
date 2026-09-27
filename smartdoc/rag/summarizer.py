"""Document summarization and key-point extraction with map-reduce for long text."""

from __future__ import annotations

import re

from smartdoc.config import Settings
from smartdoc.llm.gemini_client import GeminiClient
from smartdoc.llm.prompts import (
    build_key_points_prompt,
    build_reduce_prompt,
    build_summary_prompt,
)
from smartdoc.logging_setup import get_logger
from smartdoc.models import DocumentInsights, TextChunk

logger = get_logger()

_MAP_CHAR_BUDGET = 8000


class DocumentSummarizer:
    def __init__(self, settings: Settings, llm: GeminiClient | None = None) -> None:
        self.settings = settings
        self.llm = llm or GeminiClient(settings.gemini_api_key, settings.gemini_model)

    def analyze(self, chunks: list[TextChunk]) -> DocumentInsights:
        names = list(dict.fromkeys(c.document_name for c in chunks))
        groups = _grouped_text(chunks, budget=_MAP_CHAR_BUDGET)
        used_map_reduce = len(groups) > 1
        if used_map_reduce:
            logger.info("Using map-reduce summarization over %s groups", len(groups))
            partial_summaries = [
                self.llm.generate(build_summary_prompt(group, names)) for group in groups
            ]
            summary = self.llm.generate(build_reduce_prompt(partial_summaries, "summary"))
            partial_points = [
                self.llm.generate(build_key_points_prompt(group, names)) for group in groups
            ]
            points_raw = self.llm.generate(build_reduce_prompt(partial_points, "key-point list"))
        else:
            body = groups[0] if groups else ""
            summary = self.llm.generate(build_summary_prompt(body, names))
            points_raw = self.llm.generate(build_key_points_prompt(body, names))

        key_points = _parse_bullets(points_raw)
        structured = _structured_overview(chunks, names)
        return DocumentInsights(
            summary=summary.strip(),
            key_points=key_points,
            structured_notes=structured,
            used_map_reduce=used_map_reduce,
            document_names=names,
        )


def _grouped_text(chunks: list[TextChunk], budget: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    size = 0
    for chunk in chunks:
        block = (
            f"[{chunk.document_name} | page {chunk.page_label()}]\n{chunk.text.strip()}"
        )
        if current and size + len(block) + 2 > budget:
            groups.append("\n\n".join(current))
            current = [block]
            size = len(block)
        else:
            current.append(block)
            size += len(block) + 2
    if current:
        groups.append("\n\n".join(current))
    return groups or [""]


def _parse_bullets(text: str) -> list[str]:
    points: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*([-*•]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            points.append(cleaned)
    if not points and text.strip():
        points = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return points[:12]


def _structured_overview(chunks: list[TextChunk], names: list[str]) -> str:
    by_doc: dict[str, int] = {}
    pages: dict[str, set[str]] = {}
    for chunk in chunks:
        by_doc[chunk.document_name] = by_doc.get(chunk.document_name, 0) + 1
        pages.setdefault(chunk.document_name, set()).add(chunk.page_label())
    lines = [f"Indexed documents: {len(names)}", f"Total chunks: {len(chunks)}"]
    for name in names:
        page_labels = sorted(pages.get(name, set()), key=lambda x: (x == "N/A", x))
        lines.append(
            f"- {name}: {by_doc.get(name, 0)} chunks; pages {', '.join(page_labels[:12])}"
        )
    return "\n".join(lines)
