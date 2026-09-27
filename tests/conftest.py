from __future__ import annotations

import sys
from pathlib import Path
import hashlib

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartdoc.config import load_settings
from smartdoc.embeddings.encoder import EmbeddingEncoder
from smartdoc.models import TextChunk


@pytest.fixture
def settings():
    return load_settings()


class DeterministicEncoder(EmbeddingEncoder):
    """Stable hash-based stand-in used when the real model is not under test."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__(model_name="deterministic-test")
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _load(self):
        return self

    def encode(self, texts):
        texts = list(texts)
        vectors = np.zeros((len(texts), self._dim), dtype="float32")

        for i, text in enumerate(texts):
            tokens = text.lower().split()

            for token in tokens:
                digest = hashlib.sha256(
                    token.encode("utf-8")
                ).digest()

                slot = int.from_bytes(
                    digest[:8],
                    "little"
                ) % self._dim

                vectors[i, slot] += 1.0

            norm = np.linalg.norm(vectors[i])

            if norm:
                vectors[i] /= norm

        return vectors


def chunk(
    text: str,
    name: str = "doc.txt",
    page: int | None = 1,
    index: int = 0,
) -> TextChunk:
    return TextChunk(
        chunk_id=f"{name}-{index}",
        document_name=name,
        file_type=Path(name).suffix.lstrip(".") or "txt",
        text=text,
        chunk_index=index,
        page_number=page,
    )