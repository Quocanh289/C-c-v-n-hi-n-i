"""YouTube comments — Data API v3 (preferred) or Playwright fallback (no API key)."""

import re
from typing import AsyncIterator
from urllib.parse import quote

from data_engine.core.base_crawler import BaseCrawler
from data_engine.core.models import CrawlJob, Platform, RawPost
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)

VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"
)
VIDEO_ID_ONLY = re.compile(r"^[a-zA-Z0-9_-]{11}$")

COMMENT_SELECTORS = [
    "ytd-comment-thread-renderer #content-text",
    "ytd-comment-renderer #content-text",
    "#content-text",
]


def extract_video_id(target: str) -> str:
    """Parse video ID from URL or bare id."""
    target = target.strip()
    match = VIDEO_ID_RE.search(target)
    if match:
        return match.group(1)
    if VIDEO_ID_ONLY.match(target):
        return target
    # Last resort: v= param
    if "v=" in target:
        return target.split("v=")[1].split("&")[0][:11]
    return target


class YouTubeCrawler(BaseCrawler):
    platform = Platform.YOUTUBE
    rate_limit_rpm = 100

    async def discover(self, job: CrawlJob) -> AsyncIterator[str]:
        raw = job.target or job.params.get("video_id", "")
        if raw:
            yield extract_video_id(raw)
            return

        query = job.params.get("query", "việt nam vlog")
        if not self.settings.youtube_api_key:
            raise ValueError(
                "Search requires YOUTUBE_API_KEY. Pass a video URL or id instead:\n"
                '  --target "https://www.youtube.com/watch?v=VIDEO_ID"'
            )

        url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=snippet&q={quote(query)}&type=video&maxResults=20"
            f"&relevanceLanguage=vi&key={self.settings.youtube_api_key}"
        )
        resp = await self.http_get(url)
        data = await resp.json()

        for item in data.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                yield vid

    async def fetch_posts(self, target: str, job: CrawlJob) -> AsyncIterator[RawPost]:
        video_id = extract_video_id(target)
        limit = job.params.get("limit", 100)

        if self.settings.youtube_api_key:
            async for post in self._fetch_via_api(video_id, job, limit):
                yield post
            return

        logger.info(
            "youtube_playwright_fallback",
            video_id=video_id,
            msg="No YOUTUBE_API_KEY — scraping comments via browser",
        )
        async for post in self._fetch_via_playwright(video_id, job, limit):
            yield post

    async def _fetch_via_api(
        self,
        video_id: str,
        job: CrawlJob,
        limit: int,
    ) -> AsyncIterator[RawPost]:
        page_token = None
        collected = 0
        max_pages = job.params.get("max_pages", 10)

        for _ in range(max_pages):
            if collected >= limit:
                break

            url = (
                "https://www.googleapis.com/youtube/v3/commentThreads"
                f"?part=snippet&videoId={video_id}&maxResults=100"
                f"&textFormat=plainText&key={self.settings.youtube_api_key}"
            )
            if page_token:
                url += f"&pageToken={page_token}"

            resp = await self.http_get(url)
            data = await resp.json()

            if "error" in data:
                err = data["error"].get("message", data["error"])
                logger.warning("youtube_api_error", error=err)
                if collected == 0:
                    async for post in self._fetch_via_playwright(video_id, job, limit):
                        yield post
                return

            for item in data.get("items", []):
                if collected >= limit:
                    return
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                text = snippet.get("textDisplay", "").strip()
                if not text:
                    continue

                collected += 1
                yield RawPost(
                    platform=Platform.YOUTUBE,
                    text=text,
                    post_id=item.get("id"),
                    author_id=snippet.get("authorChannelId", {}).get("value"),
                    url=f"https://youtube.com/watch?v={video_id}",
                    language_hint=snippet.get("viewerRating"),
                    metadata={
                        "video_id": video_id,
                        "like_count": snippet.get("likeCount"),
                        "published_at": snippet.get("publishedAt"),
                        "source": "api",
                    },
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    async def _fetch_via_playwright(
        self,
        video_id: str,
        job: CrawlJob,
        limit: int,
    ) -> AsyncIterator[RawPost]:
        url = f"https://www.youtube.com/watch?v={video_id}"
        max_scrolls = job.params.get("max_scrolls", max(15, limit // 5))

        await self.rate_limiter.wait_for_slot(self.platform.value, self.rate_limit_rpm)

        async with self.browser.acquire_page(locale="vi-VN") as page:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)

            # Scroll to load comment section
            for _ in range(max_scrolls):
                await page.evaluate("window.scrollBy(0, 900)")
                await page.wait_for_timeout(1200)

            seen: set[str] = set()
            collected = 0

            for selector in COMMENT_SELECTORS:
                loc = page.locator(selector)
                count = await loc.count()
                for i in range(count):
                    if collected >= limit:
                        return
                    try:
                        text = (await loc.nth(i).inner_text()).strip()
                    except Exception:
                        continue
                    text = re.sub(r"\s+", " ", text)
                    if len(text) < 2 or text in seen:
                        continue
                    seen.add(text)
                    collected += 1

                    yield RawPost(
                        platform=Platform.YOUTUBE,
                        text=text,
                        post_id=f"{video_id}_{i}",
                        url=url,
                        metadata={"video_id": video_id, "source": "playwright", "index": i},
                    )

                if collected >= limit // 2:
                    break

            logger.info("youtube_playwright_done", video_id=video_id, collected=collected)
