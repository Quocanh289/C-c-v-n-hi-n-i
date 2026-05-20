"""In-memory backends for local dev when Redis is not running."""

import asyncio
import pickle
import time
from collections import defaultdict
from typing import Optional

from datasketch import MinHash

from data_engine.config.settings import get_settings
from data_engine.core.dedup import build_minhash, minhash_bucket_key, text_hash
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class LocalRateLimiter:
    """Process-local rate limiter (same interface as DistributedRateLimiter)."""

    def __init__(self):
        self.settings = get_settings()
        self._counts: dict[str, int] = defaultdict(int)
        self._window_start: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        platform: str,
        limit_rpm: Optional[int] = None,
        window_sec: int = 60,
    ) -> bool:
        limit = limit_rpm or self.settings.default_rate_limit_rpm
        now = time.monotonic()

        async with self._lock:
            start = self._window_start.get(platform, now)
            if now - start >= window_sec:
                self._window_start[platform] = now
                self._counts[platform] = 0

            self._counts[platform] += 1
            return self._counts[platform] <= limit

    async def wait_for_slot(
        self,
        platform: str,
        limit_rpm: Optional[int] = None,
        max_wait_sec: float = 120.0,
    ) -> None:
        start = time.monotonic()
        while True:
            if await self.acquire(platform, limit_rpm):
                return
            if time.monotonic() - start > max_wait_sec:
                raise TimeoutError(f"Rate limit wait exceeded for {platform}")
            await asyncio.sleep(0.5)


class InMemoryDedupStore:
    """In-memory dedup (exact + MinHash) for local CLI runs."""

    def __init__(self):
        self.settings = get_settings()
        self._exact: set[str] = set()
        self._minhash_buckets: dict[str, list[bytes]] = defaultdict(list)

    async def is_exact_duplicate(self, text: str) -> bool:
        h = text_hash(text)
        if h in self._exact:
            return True
        self._exact.add(h)
        return False

    async def is_near_duplicate(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> bool:
        threshold = threshold or self.settings.minhash_threshold
        mh = build_minhash(text)
        bucket = minhash_bucket_key(mh)

        for raw in self._minhash_buckets[bucket][:50]:
            other = pickle.loads(raw)
            if mh.jaccard(other) >= threshold:
                return True

        self._minhash_buckets[bucket].insert(0, pickle.dumps(mh))
        self._minhash_buckets[bucket] = self._minhash_buckets[bucket][:100]
        return False

    async def is_duplicate(
        self,
        text: str,
        check_near: bool = True,
    ) -> tuple[bool, str]:
        if await self.is_exact_duplicate(text):
            return True, "exact"
        if check_near and await self.is_near_duplicate(text):
            return True, "near"
        return False, "none"
