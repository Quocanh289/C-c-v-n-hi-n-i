"""Facebook public pages/posts — browser automation only."""

import re
from typing import AsyncIterator

from data_engine.core.base_crawler import BaseCrawler
from data_engine.core.models import CrawlJob, Platform, RawPost
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)

# UI / login chrome — not post content
NOISE_EXACT = {
    "giới thiệu", "đang theo dõi", "theo dõi", "quên mật khẩu?",
    "đăng nhập", "đăng ký", "tạo trang", "xem thêm", "see more",
    "forgot password?", "log in", "sign up", "follow", "about",
}
NOISE_PATTERN = re.compile(
    r"(?i)(quên mật khẩu|đăng nhập|log in|sign up|@gmail\.com|@facebook\.com|"
    r"xem thêm từ|see more from|^\d+[\s,]*người theo dõi)",
)


def resolve_facebook_url(target: str) -> str:
    """Convert slug or full URL to a Facebook page URL."""
    target = target.strip()
    if target.startswith("http://") or target.startswith("https://"):
        return target
    slug = target.lstrip("@/")
    return f"https://www.facebook.com/{slug}"


def is_ui_noise(text: str) -> bool:
    lower = text.lower().strip()
    if lower in NOISE_EXACT:
        return True
    if NOISE_PATTERN.search(text):
        return True
    if re.fullmatch(r"[\w.+-]+@[\w-]+\.\w+", text):
        return True
    return False


class FacebookCrawler(BaseCrawler):
    """
    Crawls PUBLIC Facebook page content via Playwright.
    Guest/headless sessions see limited feed — expect far fewer posts than --limit.
    """

    platform = Platform.FACEBOOK
    rate_limit_rpm = 10
    use_browser = True

    # Fallback regex when structured selectors miss
    COMMENT_PATTERN = re.compile(
        r'dir="auto"[^>]*>([^<]{15,2000})</span>',
        re.DOTALL,
    )

    POST_SELECTORS = [
        '[data-ad-preview="message"]',
        'div[data-ad-comet-preview="message"]',
        'div[role="article"] div[dir="auto"]',
        'div[role="feed"] div[dir="auto"]',
    ]

    async def discover(self, job: CrawlJob) -> AsyncIterator[str]:
        raw = job.target or job.params.get("page_url")
        if not raw:
            raise ValueError(
                "Facebook crawler requires a public page URL or page slug.\n"
                "  Example: --target https://www.facebook.com/VietNam.official\n"
                "  Example: --target VietNam.official"
            )
        page_url = resolve_facebook_url(raw)
        logger.info("facebook_target_resolved", raw=raw, url=page_url)
        yield page_url

    async def _extract_via_locators(self, page, limit: int, seen: set[str]) -> list[str]:
        texts: list[str] = []
        for selector in self.POST_SELECTORS:
            loc = page.locator(selector)
            count = await loc.count()
            for i in range(min(count, limit * 3)):
                if len(texts) >= limit:
                    break
                try:
                    raw = (await loc.nth(i).inner_text()).strip()
                except Exception:
                    continue
                text = re.sub(r"\s+", " ", raw)
                if len(text) < 15 or text in seen or is_ui_noise(text):
                    continue
                seen.add(text)
                texts.append(text)
            if len(texts) >= limit // 2:
                break
        return texts

    async def _extract_via_regex(self, content: str, limit: int, seen: set[str]) -> list[str]:
        texts: list[str] = []
        for raw in self.COMMENT_PATTERN.findall(content):
            if len(texts) >= limit:
                break
            text = re.sub(r"<[^>]+>", "", raw)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 15 or text in seen or is_ui_noise(text):
                continue
            seen.add(text)
            texts.append(text)
        return texts

    async def fetch_posts(self, target: str, job: CrawlJob) -> AsyncIterator[RawPost]:
        limit = job.params.get("limit", 50)
        # More scrolls when user asks for more posts
        max_scrolls = job.params.get("max_scrolls", max(15, min(50, limit // 5)))

        await self.rate_limiter.wait_for_slot(self.platform.value, self.rate_limit_rpm)

        async with self.browser.acquire_page(locale="vi-VN") as page:
            await page.goto(target, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(4000)

            # Dismiss cookie/login overlays if present (best-effort)
            for dismiss_sel in (
                '[aria-label="Close"]',
                '[aria-label="Đóng"]',
            ):
                btn = page.locator(dismiss_sel).first
                if await btn.count() > 0:
                    try:
                        await btn.click(timeout=2000)
                        await page.wait_for_timeout(1000)
                    except Exception:
                        pass

            scrolls = await self.browser.infinite_scroll(page, max_scrolls=max_scrolls)
            logger.info("facebook_scrolled", scrolls=scrolls, max_scrolls=max_scrolls)

            seen: set[str] = set()
            texts = await self._extract_via_locators(page, limit, seen)
            if len(texts) < limit // 3:
                content = await page.content()
                texts.extend(await self._extract_via_regex(content, limit, seen))

            logger.info(
                "facebook_extracted",
                raw_blocks=len(texts),
                limit=limit,
                note="Guest headless view is limited; use logged-in session for full feed",
            )

            for idx, text in enumerate(texts[:limit]):
                yield RawPost(
                    platform=Platform.FACEBOOK,
                    text=text,
                    url=target,
                    metadata={"extraction": "playwright", "index": idx},
                )
