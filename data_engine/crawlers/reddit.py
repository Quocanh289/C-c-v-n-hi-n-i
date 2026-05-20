"""Reddit crawler — OAuth API first, browser fallback."""

import base64
from typing import AsyncIterator, Optional

from data_engine.core.base_crawler import BaseCrawler
from data_engine.core.models import CrawlJob, Platform, RawPost
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class RedditCrawler(BaseCrawler):
    platform = Platform.REDDIT
    rate_limit_rpm = 60

    def __init__(self, redis_client, **kwargs):
        super().__init__(redis_client, **kwargs)
        self._token: Optional[str] = None

    async def _get_oauth_token(self) -> str:
        if self._token:
            return self._token

        if not self.settings.reddit_client_id or not self.settings.reddit_client_secret:
            raise ValueError("Reddit API credentials not configured")

        auth = base64.b64encode(
            f"{self.settings.reddit_client_id}:{self.settings.reddit_client_secret}".encode()
        ).decode()

        headers = {
            "Authorization": f"Basic {auth}",
            "User-Agent": self.settings.reddit_user_agent or self.settings.user_agent,
        }
        data = {"grant_type": "client_credentials"}

        async with self.session.post(
            "https://www.reddit.com/api/v1/access_token",
            headers=headers,
            data=data,
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
            self._token = payload["access_token"]
            return self._token

    async def discover(self, job: CrawlJob) -> AsyncIterator[str]:
        subreddit = job.target or job.params.get("subreddit", "vietnam")
        yield subreddit

    async def fetch_posts(self, target: str, job: CrawlJob) -> AsyncIterator[RawPost]:
        sort = job.params.get("sort", "hot")
        limit = min(job.params.get("limit", 100), 100)

        has_api = bool(
            self.settings.reddit_client_id and self.settings.reddit_client_secret
        )
        if not has_api:
            logger.info("reddit_public_json", subreddit=target)
            async for post in self._fetch_public_json(target, sort, limit):
                yield post
            return

        try:
            token = await self._get_oauth_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": self.settings.reddit_user_agent or self.settings.user_agent,
            }
            url = (
                f"https://oauth.reddit.com/r/{target}/{sort}.json"
                f"?limit={limit}&raw_json=1"
            )
            resp = await self.http_get(url, headers=headers)
            data = await resp.json()

            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                text = f"{title}\n{selftext}".strip() if selftext else title

                if not text:
                    continue

                yield RawPost(
                    platform=Platform.REDDIT,
                    text=text,
                    post_id=post.get("id"),
                    author_id=post.get("author"),
                    url=f"https://reddit.com{post.get('permalink', '')}",
                    metadata={
                        "subreddit": target,
                        "score": post.get("score"),
                        "num_comments": post.get("num_comments"),
                        "sort": sort,
                    },
                )

                # Fetch top comments if requested
                if job.params.get("include_comments") and post.get("num_comments", 0) > 0:
                    async for comment in self._fetch_comments(target, post["id"], headers):
                        yield comment

        except Exception:
            logger.warning("reddit_api_unavailable", msg="Falling back to public JSON")
            async for post in self._fetch_public_json(target, sort, limit):
                yield post

    async def _fetch_comments(
        self,
        subreddit: str,
        post_id: str,
        headers: dict,
    ) -> AsyncIterator[RawPost]:
        url = f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}.json?limit=50&raw_json=1"
        resp = await self.http_get(url, headers=headers)
        data = await resp.json()

        if len(data) < 2:
            return

        comments = data[1].get("data", {}).get("children", [])
        for child in comments:
            c = child.get("data", {})
            body = c.get("body", "")
            if body and body not in ("[deleted]", "[removed]"):
                yield RawPost(
                    platform=Platform.REDDIT,
                    text=body,
                    post_id=c.get("id"),
                    parent_id=post_id,
                    author_id=c.get("author"),
                    metadata={"type": "comment", "subreddit": subreddit},
                )

    async def _fetch_public_json(
        self,
        subreddit: str,
        sort: str,
        limit: int,
    ) -> AsyncIterator[RawPost]:
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}&raw_json=1"
        resp = await self.http_get(url)
        data = await resp.json()

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            text = post.get("title", "")
            if text:
                yield RawPost(
                    platform=Platform.REDDIT,
                    text=text,
                    post_id=post.get("id"),
                    url=f"https://reddit.com{post.get('permalink', '')}",
                    metadata={"subreddit": subreddit, "fallback": True},
                )
