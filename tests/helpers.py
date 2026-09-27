"""In-memory upload objects and synthetic PDF/DOCX/TXT builders."""

from __future__ import annotations

import io
from pathlib import Path

from docx import Document
from pypdf import PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject


class MemoryUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data
        self.size = len(data)

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


def make_txt(name: str, text: str) -> MemoryUpload:
    return MemoryUpload(name, text.encode("utf-8"))


def make_docx(name: str, paragraphs: list[str]) -> MemoryUpload:
    document = Document()
    for para in paragraphs:
        document.add_paragraph(para)
    buffer = io.BytesIO()
    document.save(buffer)
    return MemoryUpload(name, buffer.getvalue())


def make_pdf(name: str, pages: list[str]) -> MemoryUpload:
    """Create a simple multi-page PDF with visible text per page."""
    writer = PdfWriter()
    for text in pages:
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[-1]
        _add_text(page, text)
    buffer = io.BytesIO()
    writer.write(buffer)
    return MemoryUpload(name, buffer.getvalue())


def _add_text(page, text: str) -> None:
    """Attach a content stream so pypdf can extract the text later."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    lines = safe.split("\n")
    commands = ["BT", "/F1 12 Tf", "72 740 Td"]
    for i, line in enumerate(lines):
        if i:
            commands.append("0 -16 Td")
        commands.append(f"({line[:110]}) Tj")
    commands.append("ET")
    content = "\n".join(commands).encode("latin-1", errors="replace")
    stream = DecodedStreamObject()
    stream.set_data(content)
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    resources = page.get("/Resources")
    if resources is None:
        resources = DictionaryObject()
        page[NameObject("/Resources")] = resources
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = font
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Contents")] = stream


POLICY_TEXT = (
    "Aether Labs remote work policy states that full-time engineers may work remotely "
    "three days per week. Core collaboration hours are 11:00 to 16:00 IST. "
    "Employees must attend the Monday architecture review in person."
)

SAFETY_TEXT = (
    "Laboratory safety requires closed-toe shoes, eye protection, and a buddy system "
    "after 20:00. Chemical waste must be logged in the yellow register."
)

UNKNOWN_QUERY = "What is the capital of Atlantis according to these files?"


def sample_corpus_files() -> list[MemoryUpload]:
    return [
        make_pdf("policy.pdf", [POLICY_TEXT, "Page two notes that laptops must use disk encryption."]),
        make_docx(
            "safety.docx",
            [SAFETY_TEXT, "First-aid kits are stored near the south stairwell."],
        ),
        make_txt(
            "handbook.txt",
            "The internship handbook explains that weekly reports are due every Friday by 17:00.",
        ),
    ]


def write_samples(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files = sample_corpus_files()
    for item in files:
        (directory / item.name).write_bytes(item.getvalue())
