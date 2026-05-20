"""Abstract base crawler — all platform crawlers inherit from this."""

from abc import ABC, abstractmethod
import asyncio
from typing import AsyncIterator, Optional

import aiohttp
import redis.asyncio as aioredis

from data_engine.config.settings import get_settings
from data_engine.core.browser_session import BrowserPool
from data_engine.core.dedup import DedupStore
from data_engine.core.models import CrawlJob, CrawlStatus, Platform, RawPost
from data_engine.core.queue import CrawlQueue
from data_engine.core.rate_limiter import DistributedRateLimiter
from data_engine.core.retry import with_retry
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class BaseCrawler(ABC):
    """
    Platform crawler contract:
    - discover() yields seed URLs/targets
    - crawl() fetches and yields RawPost items
    - Integrates rate limiting, retry, dedup at framework level
    """

    platform: Platform
    rate_limit_rpm: int = 30
    use_browser: bool = False

    def __init__(
        self,
        redis_client: Optional[aioredis.Redis] = None,
        session: Optional[aiohttp.ClientSession] = None,
        browser_pool: Optional[BrowserPool] = None,
        local_mode: bool = False,
    ):
        self.settings = get_settings()
        self.local_mode = local_mode or redis_client is None
        self.redis = redis_client

        if self.local_mode:
            from data_engine.core.local_backend import InMemoryDedupStore, LocalRateLimiter

            self.queue = None
            self.rate_limiter = LocalRateLimiter()
            self.dedup = InMemoryDedupStore()
            logger.info("crawler_local_mode", platform=self.platform.value)
        else:
            self.queue = CrawlQueue(redis_client)
            self.rate_limiter = DistributedRateLimiter(redis_client)
            self.dedup = DedupStore(redis_client)

        self._session = session
        self._browser_pool = browser_pool
        self._owns_session = session is None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.settings.request_timeout_sec)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": self.settings.user_agent},
            )
            self._owns_session = True
        return self._session

    @property
    def browser(self) -> BrowserPool:
        if self._browser_pool is None:
            self._browser_pool = BrowserPool()
        return self._browser_pool

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
            # Brief yield so Windows proactor SSL transport can shut down cleanly
            await asyncio.sleep(0.25)
        if self._browser_pool:
            await self._browser_pool.stop()

    @abstractmethod
    async def discover(self, job: CrawlJob) -> AsyncIterator[str]:
        """Yield seed targets (URLs, IDs) for this job."""
        yield ""  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    async def fetch_posts(self, target: str, job: CrawlJob) -> AsyncIterator[RawPost]:
        """Fetch raw posts for a single target."""
        yield RawPost(platform=self.platform, text="")  # pragma: no cover
        raise NotImplementedError

    async def http_get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        await self.rate_limiter.wait_for_slot(
            self.platform.value,
            self.rate_limit_rpm,
        )

        async def _request():
            return await self.session.get(url, **kwargs)

        return await with_retry(_request, operation_name=f"{self.platform.value}_get")

    async def run_job(self, job: CrawlJob) -> AsyncIterator[RawPost]:
        """Execute full crawl job with dedup filtering; yields non-duplicate posts."""
        job.status = CrawlStatus.RUNNING
        collected = 0
        skipped = 0

        logger.info(
            "job_started",
            job_id=job.id,
            platform=self.platform.value,
            target=job.target,
        )

        try:
            async for target in self.discover(job):
                if not target:
                    continue
                async for post in self.fetch_posts(target, job):
                    if len(post.text.strip()) < self.settings.min_text_length:
                        skipped += 1
                        continue

                    is_dup, reason = await self.dedup.is_duplicate(post.text)
                    if is_dup:
                        skipped += 1
                        logger.debug("skipped_duplicate", reason=reason)
                        continue

                    collected += 1
                    yield post

            job.items_collected = collected
            job.items_skipped = skipped
            job.status = CrawlStatus.COMPLETED
        except Exception as exc:
            job.status = CrawlStatus.FAILED
            job.error_message = str(exc)
            logger.exception("job_failed", job_id=job.id, error=str(exc))
            raise
        finally:
            logger.info(
                "job_finished",
                job_id=job.id,
                collected=collected,
                skipped=skipped,
            )
