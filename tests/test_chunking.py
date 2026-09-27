from smartdoc.document_processing.chunker import chunk_documents
from smartdoc.document_processing.extractor import extract_documents
from smartdoc.document_processing.preprocessor import preprocess_text
from smartdoc.models import DocumentRecord, ExtractedPage
from tests.helpers import sample_corpus_files


def test_preprocessor_normalizes_whitespace():
    raw = "Hello,\r\n\r\n\r\nworld-\nwide   team.\x0c"
    cleaned = preprocess_text(raw)
    assert "\r" not in cleaned
    assert "   " not in cleaned
    assert "worldwide" in cleaned or "world-wide" in cleaned or "world" in cleaned


def test_chunking_preserves_source_and_pages():
    doc = DocumentRecord(
        document_name="policy.pdf",
        file_type="pdf",
        size_bytes=100,
        page_count=2,
        char_count=80,
        pages=[
            ExtractedPage("policy.pdf", "pdf", "Remote work is allowed three days per week.", 1),
            ExtractedPage("policy.pdf", "pdf", "Encryption is mandatory on every laptop.", 2),
        ],
    )
    chunks = chunk_documents([doc], chunk_size=80, chunk_overlap=20)
    assert chunks
    assert all(c.document_name == "policy.pdf" for c in chunks)
    pages = {c.page_number for c in chunks}
    assert 1 in pages or 2 in pages
    assert all(c.page_label() != "" for c in chunks)


def test_chunking_multiple_real_files():
    records = extract_documents(sample_corpus_files(), max_file_size_bytes=10_000_000)
    chunks = chunk_documents(records, chunk_size=180, chunk_overlap=40)
    names = {c.document_name for c in chunks}
    assert "policy.pdf" in names
    assert "safety.docx" in names
    assert "handbook.txt" in names
    assert all(c.text.strip() for c in chunks)


def test_overlap_creates_more_than_one_chunk_for_long_text():
    long_text = " ".join([f"Sentence number {i} about internships and retrieval." for i in range(40)])
    doc = DocumentRecord(
        document_name="long.txt",
        file_type="txt",
        size_bytes=len(long_text),
        page_count=1,
        char_count=len(long_text),
        pages=[ExtractedPage("long.txt", "txt", long_text, None, 1, 40)],
    )
    chunks = chunk_documents([doc], chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
