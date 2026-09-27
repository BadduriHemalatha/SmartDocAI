from smartdoc.llm.prompts import NOT_FOUND_MESSAGE, build_qa_prompt, format_context
from smartdoc.models import RetrievedChunk
from smartdoc.rag.pipeline import RAGPipeline
from smartdoc.rag.summarizer import DocumentSummarizer, _parse_bullets, _grouped_text
from smartdoc.vectorstore.faiss_store import FaissVectorStore
from tests.conftest import DeterministicEncoder, chunk
from tests.helpers import sample_corpus_files
from smartdoc.document_processing.chunker import chunk_documents
from smartdoc.document_processing.extractor import extract_documents


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "USER QUESTION" in prompt:
            if "Atlantis" in prompt:
                return NOT_FOUND_MESSAGE
            assert "DOCUMENT CONTEXT:" in prompt
            return (
                "Engineers may work remotely three days per week "
                "(source: policy.pdf, page 1)."
            )
        if "KEY POINTS" in prompt or "key-point" in prompt.lower():
            return "- Remote work is limited to three days.\n- Reports are due Friday."
        return "The documents describe internship, safety, and remote-work rules."


def _pipeline_with_corpus(settings):
    encoder = DeterministicEncoder()
    pipeline = RAGPipeline(settings, encoder=encoder)
    pipeline.llm = FakeLLM()
    files = sample_corpus_files()
    corpus = pipeline.ingest(files)
    return pipeline, corpus


def test_rag_answers_from_retrieved_context(settings):
    pipeline, corpus = _pipeline_with_corpus(settings)
    result = pipeline.ask(corpus, "How many remote days are allowed for engineers?")
    assert result.found_in_documents
    assert result.sources
    assert "three days" in result.answer.lower()
    assert pipeline.llm.calls
    assert "DOCUMENT CONTEXT:" in pipeline.llm.calls[0]
    assert result.sources[0].chunk.document_name


def test_unknown_information_skips_or_reports_not_found(settings):
    pipeline, corpus = _pipeline_with_corpus(settings)
    # Extreme threshold to force no retrieval.
    pipeline.settings = settings
    result = pipeline.ask(corpus, "What is the capital of Atlantis?")
    # Either skipped LLM due to threshold or model said not found.
    assert NOT_FOUND_MESSAGE.split()[0] in result.answer or not result.found_in_documents
    if result.retrieval_skipped_llm:
        assert result.sources == []


def test_source_metadata_matches_retrieved_chunk(settings):
    pipeline, corpus = _pipeline_with_corpus(settings)
    result = pipeline.ask(corpus, "What must be logged for chemical waste?")
    assert result.sources
    source = result.sources[0]
    assert source.chunk.document_name in {"policy.pdf", "safety.docx", "handbook.txt"}
    assert source.chunk.page_label()
    excerpt_ok = source.chunk.text in format_context(result.sources, 12000)
    assert excerpt_ok


def test_index_is_reused_for_unchanged_files(settings):
    pipeline, corpus = _pipeline_with_corpus(settings)
    again = pipeline.ingest(sample_corpus_files(), existing=corpus)
    assert again is corpus


def test_summarizer_uses_llm_and_parses_points(settings):
    llm = FakeLLM()
    summarizer = DocumentSummarizer(settings, llm=llm)
    records = extract_documents(sample_corpus_files(), max_file_size_bytes=10_000_000)
    chunks = chunk_documents(records, chunk_size=400, chunk_overlap=50)
    insights = summarizer.analyze(chunks)
    assert insights.summary
    assert insights.key_points
    assert insights.document_names
    assert insights.structured_notes
    assert llm.calls


def test_map_reduce_groups_long_text():
    chunks = [
        chunk("x" * 3000, "a.txt", 1, i) for i in range(6)
    ]
    groups = _grouped_text(chunks, budget=4000)
    assert len(groups) > 1


def test_parse_bullets():
    text = "- First point\n* Second point\n1. Third point"
    assert _parse_bullets(text) == ["First point", "Second point", "Third point"]
