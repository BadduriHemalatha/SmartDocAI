"""Sentence Transformer embedding generation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from smartdoc.exceptions import EmbeddingError
from smartdoc.logging_setup import get_logger

logger = get_logger()


class EmbeddingEncoder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "sentence-transformers is not installed",
                user_message="The embedding library is missing. Install dependencies from requirements.txt.",
            ) from exc
        try:
            logger.info("Loading embedding model %s", self.model_name)
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except Exception as exc:
            logger.exception("Failed to load embedding model")
            raise EmbeddingError(
                f"Failed to load embedding model {self.model_name}: {exc}",
                user_message=(
                    "Could not load the Sentence Transformer model. "
                    "Check your internet connection on first run (model download)."
                ),
            ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        dim = model.get_sentence_embedding_dimension()
        if not dim:
            raise EmbeddingError("Embedding model did not report a dimension.")
        return int(dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError(
                "No texts to embed",
                user_message="There is no text available to generate embeddings.",
            )
        try:
            model = self._load()
            vectors = model.encode(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise EmbeddingError(
                f"Embedding failed: {exc}",
                user_message="Failed to generate semantic embeddings for the documents.",
            ) from exc
        array = np.asarray(vectors, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array
