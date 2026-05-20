"""Sentence embedding encoder for semantic dedup and slang clustering."""

from functools import lru_cache
from typing import Optional

import numpy as np

from data_engine.config.settings import get_settings
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class EmbeddingEncoder:
    """Lazy-loaded sentence-transformers encoder."""

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()
