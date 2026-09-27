from smartdoc.document_processing.extractor import extract_documents, extract_one
from smartdoc.exceptions import (
    CorruptedFileError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from tests.helpers import MemoryUpload, make_docx, make_pdf, make_txt, sample_corpus_files


def test_pdf_extraction_keeps_page_numbers():
    upload = make_pdf("policy.pdf", ["Alpha page one content.", "Beta page two content."])
    record = extract_one(upload, max_file_size_bytes=10_000_000)
    assert record.file_type == "pdf"
    assert record.page_count == 2
    assert record.pages[0].page_number == 1
    assert record.pages[1].page_number == 2
    assert "Alpha" in record.pages[0].text or "Alpha" in " ".join(p.text for p in record.pages)


def test_docx_extraction():
    upload = make_docx("notes.docx", ["Internship goals include retrieval augmented generation."])
    record = extract_one(upload, max_file_size_bytes=10_000_000)
    assert record.file_type == "docx"
    assert "retrieval" in record.pages[0].text.lower()
    assert record.pages[0].page_number is None


def test_txt_extraction_and_line_range():
    upload = make_txt("readme.txt", "Line one\nLine two\nLine three")
    record = extract_one(upload, max_file_size_bytes=10_000_000)
    assert record.file_type == "txt"
    assert record.pages[0].start_line == 1
    assert record.pages[0].end_line == 3
    assert "Line two" in record.pages[0].text


def test_multiple_documents():
    records = extract_documents(sample_corpus_files(), max_file_size_bytes=10_000_000)
    names = {r.document_name for r in records}
    assert names == {"policy.pdf", "safety.docx", "handbook.txt"}
    assert all(r.char_count > 0 for r in records)


def test_unsupported_file_type():
    upload = MemoryUpload("image.png", b"not-a-document")
    try:
        extract_one(upload, max_file_size_bytes=10_000_000)
        assert False, "expected error"
    except UnsupportedFileTypeError as exc:
        assert "supported" in exc.user_message.lower()


def test_empty_file():
    upload = MemoryUpload("empty.txt", b"")
    try:
        extract_one(upload, max_file_size_bytes=10_000_000)
        assert False, "expected error"
    except EmptyDocumentError:
        pass


def test_file_too_large():
    upload = make_txt("big.txt", "hello")
    try:
        extract_one(upload, max_file_size_bytes=1)
        assert False, "expected error"
    except FileTooLargeError:
        pass


def test_corrupted_pdf():
    upload = MemoryUpload("broken.pdf", b"%PDF-1.4 not really a pdf")
    try:
        extract_one(upload, max_file_size_bytes=10_000_000)
        assert False, "expected error"
    except (CorruptedFileError, EmptyDocumentError):
        pass
