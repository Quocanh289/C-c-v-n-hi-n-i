"""Multi-layer deduplication: exact hash, MinHash, semantic (Qdrant)."""

import hashlib
import pickle
import re
from typing import Optional

import redis.asyncio as aioredis
from datasketch import MinHash

from data_engine.config.settings import get_settings
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


def normalize_for_hash(text: str) -> str:
    """Light normalization for dedup only — NOT for training text."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def text_hash(text: str) -> str:
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    tokens = normalize_for_hash(text).split()
    for token in tokens:
        mh.update(token.encode("utf-8"))
    for bigram in zip(tokens, tokens[1:]):
        mh.update(" ".join(bigram).encode("utf-8"))
    return mh


def minhash_bucket_key(mh: MinHash, prefix_len: int = 8) -> str:
    """LSH bucket key — compatible with datasketch digest() as bytes or numpy array."""
    digest = mh.digest()
    chunk = digest[:prefix_len]
    if hasattr(chunk, "tobytes"):
        return chunk.tobytes().hex()
    return bytes(chunk).hex()


class DedupStore:
    """
    Layer 1: Redis SET for exact SHA256 hashes (O(1) lookup).
    Layer 2: MinHash LSH buckets in Redis for near-duplicates.
    Layer 3: Semantic dedup via Qdrant (handled in embeddings module).
    """

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.settings = get_settings()
        self.exact_key = "dedup:exact"
        self.minhash_prefix = "dedup:minhash:"

    async def is_exact_duplicate(self, text: str) -> bool:
        h = text_hash(text)
        added = await self.redis.sadd(self.exact_key, h)
        return added == 0

    async def is_near_duplicate(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> bool:
        """Check MinHash similarity against recent bucket."""
        threshold = threshold or self.settings.minhash_threshold
        mh = build_minhash(text)
        bucket = minhash_bucket_key(mh)
        key = f"{self.minhash_prefix}{bucket}"

        stored = await self.redis.lrange(key, 0, 50)
        for item in stored:
            raw = item if isinstance(item, bytes) else item.encode()
            other = pickle.loads(raw)
            if mh.jaccard(other) >= threshold:
                return True

        await self.redis.lpush(key, pickle.dumps(mh))
        await self.redis.ltrim(key, 0, 100)
        await self.redis.expire(key, 86400 * 7)
        return False

    async def is_duplicate(
        self,
        text: str,
        check_near: bool = True,
    ) -> tuple[bool, str]:
        """
        Returns (is_duplicate, reason).
        reason: exact | near | none
        """
        if await self.is_exact_duplicate(text):
            return True, "exact"
        if check_near and await self.is_near_duplicate(text):
            return True, "near"
        return False, "none"
