"""Qdrant vector store for semantic dedup and slang clustering."""

import uuid
from typing import Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from data_engine.config.settings import get_settings
from data_engine.embeddings.encoder import EmbeddingEncoder
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class QdrantVectorStore:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        encoder: Optional[EmbeddingEncoder] = None,
    ):
        self.settings = get_settings()
        self.client = client or QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )
        self.encoder = encoder or EmbeddingEncoder()
        self.collection = self.settings.qdrant_collection

    def ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.settings.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", collection=self.collection)

    def is_semantic_duplicate(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """Search for near-duplicate by cosine similarity."""
        threshold = threshold or self.settings.semantic_dedup_threshold
        vector = self.encoder.encode_single(text)

        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=1,
            score_threshold=threshold,
        )

        if results:
            return True, results[0].payload.get("text_hash")
        return False, None

    def upsert(
        self,
        text: str,
        text_hash: str,
        payload: dict[str, Any],
    ) -> str:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text_hash))
        vector = self.encoder.encode_single(text)

        self.client.upsert(
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text_hash": text_hash,
                        "text_preview": text[:200],
                        **payload,
                    },
                )
            ],
        )
        return point_id

    def cluster_slang_terms(
        self,
        terms: list[str],
        min_cluster_size: int = 3,
    ) -> list[list[str]]:
        """Group semantically similar slang candidates."""
        if len(terms) < min_cluster_size:
            return []

        vectors = self.encoder.encode(terms)
        # Simple greedy clustering by cosine similarity
        used = set()
        clusters: list[list[str]] = []
        threshold = 0.75

        for i, term in enumerate(terms):
            if i in used:
                continue
            cluster = [term]
            used.add(i)
            for j in range(i + 1, len(terms)):
                if j in used:
                    continue
                sim = float(np.dot(vectors[i], vectors[j]))
                if sim >= threshold:
                    cluster.append(terms[j])
                    used.add(j)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters
