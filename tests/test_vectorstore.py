import numpy as np

from smartdoc.vectorstore.faiss_store import FaissVectorStore
from tests.conftest import DeterministicEncoder, chunk


def test_faiss_index_and_semantic_retrieval():
    encoder = DeterministicEncoder()
    chunks = [
        chunk("Remote work policy allows three days from home.", "policy.pdf", 1, 0),
        chunk("Chemical waste must be logged in the yellow register.", "safety.docx", None, 1),
        chunk("Weekly reports are due Friday by 17:00.", "handbook.txt", None, 2),
    ]
    embeddings = encoder.encode([c.text for c in chunks])
    store = FaissVectorStore()
    store.build(embeddings, chunks)
    assert store.is_ready
    assert store.size == 3

    query = encoder.encode(["When is the weekly report due?"])
    hits = store.search(query, top_k=2)
    assert hits
    top_id, score = hits[0]
    assert score > 0
    assert "Friday" in store.get_chunk(top_id).text or "report" in store.get_chunk(top_id).text.lower()


def test_faiss_rejects_empty_embeddings():
    store = FaissVectorStore()
    try:
        store.build(np.zeros((0, 8), dtype="float32"), [])
        assert False, "expected error"
    except Exception as exc:
        assert "embed" in str(exc).lower() or "index" in str(exc).lower() or hasattr(exc, "user_message")
