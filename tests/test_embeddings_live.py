"""Optional live Sentence Transformer embedding smoke test."""

import pytest

from smartdoc.embeddings.encoder import EmbeddingEncoder
from smartdoc.exceptions import EmbeddingError


def test_sentence_transformer_embeddings_if_available():
    try:
        encoder = EmbeddingEncoder("sentence-transformers/all-MiniLM-L6-v2")
        vectors = encoder.encode(["SmartDoc AI uses retrieval augmented generation."])
    except EmbeddingError:
        pytest.skip("Sentence Transformer model is not available in this environment")
    except Exception as exc:
        pytest.skip(f"Embedding backend unavailable: {exc}")
    assert vectors.shape[0] == 1
    assert vectors.shape[1] >= 8
