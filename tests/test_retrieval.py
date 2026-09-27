from smartdoc.exceptions import InvalidQuestionError
from smartdoc.retrieval.retriever import SemanticRetriever
from smartdoc.vectorstore.faiss_store import FaissVectorStore
from tests.conftest import DeterministicEncoder, chunk


def _indexed():
    encoder = DeterministicEncoder()
    chunks = [
        chunk("Aether Labs engineers may work remotely three days per week.", "policy.pdf", 1, 0),
        chunk("Closed-toe shoes are required in the laboratory.", "safety.docx", None, 1),
    ]
    store = FaissVectorStore()
    store.build(encoder.encode([c.text for c in chunks]), chunks)
    return encoder, store


def test_retriever_returns_relevant_policy_chunk():
    encoder, store = _indexed()
    retriever = SemanticRetriever(store, encoder, top_k=2, min_similarity=0.05)
    hits = retriever.retrieve("How many remote days are allowed?")
    assert hits
    assert hits[0].rank == 1
    assert hits[0].chunk.document_name == "policy.pdf"
    assert hits[0].chunk.page_number == 1
    assert hits[0].score >= 0.05


def test_retriever_rejects_empty_question():
    encoder, store = _indexed()
    retriever = SemanticRetriever(store, encoder, top_k=2, min_similarity=0.1)
    try:
        retriever.retrieve("  ")
        assert False, "expected error"
    except InvalidQuestionError:
        pass


def test_low_similarity_yields_no_chunks():
    encoder, store = _indexed()
    retriever = SemanticRetriever(store, encoder, top_k=2, min_similarity=0.99)
    hits = retriever.retrieve("Who won the 1998 world cup in France?")
    assert hits == []
