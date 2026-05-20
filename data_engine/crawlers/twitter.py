"""X/Twitter crawler — API v2 when bearer token available."""

from typing import AsyncIterator
from urllib.parse import quote

from data_engine.core.base_crawler import BaseCrawler
from data_engine.core.models import CrawlJob, Platform, RawPost
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class TwitterCrawler(BaseCrawler):
    platform = Platform.TWITTER
    rate_limit_rpm = 50

    async def discover(self, job: CrawlJob) -> AsyncIterator[str]:
        query = job.target or job.params.get("query", "lang:vi -is:retweet")
        yield query

    async def fetch_posts(self, target: str, job: CrawlJob) -> AsyncIterator[RawPost]:
        if not self.settings.twitter_bearer_token:
            logger.warning("twitter_token_missing", msg="Configure TWITTER_BEARER_TOKEN")
            return

        max_results = min(job.params.get("limit", 100), 100)
        query_encoded = quote(target)

        url = (
            "https://api.twitter.com/2/tweets/search/recent"
            f"?query={query_encoded}&max_results={max_results}"
            "&tweet.fields=created_at,lang,public_metrics,author_id"
        )
        headers = {"Authorization": f"Bearer {self.settings.twitter_bearer_token}"}

        resp = await self.http_get(url, headers=headers)
        data = await resp.json()

        for tweet in data.get("data", []):
            text = tweet.get("text", "").strip()
            if not text or text.startswith("RT @"):
                continue

            yield RawPost(
                platform=Platform.TWITTER,
                text=text,
                post_id=tweet.get("id"),
                author_id=tweet.get("author_id"),
                url=f"https://twitter.com/i/web/status/{tweet.get('id')}",
                language_hint=tweet.get("lang"),
                metadata={
                    "metrics": tweet.get("public_metrics"),
                    "created_at": tweet.get("created_at"),
                    "query": target,
                },
            )
