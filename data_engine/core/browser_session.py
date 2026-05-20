"""Playwright browser pool for dynamic-content platforms."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from data_engine.config.settings import get_settings
from data_engine.core.playwright_check import PlaywrightNotInstalledError, install_hint
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class BrowserPool:
    """Singleton-style browser pool with context reuse."""

    def __init__(self, max_contexts: int = 4):
        self.max_contexts = max_contexts
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._launch_mode: Optional[str] = None
        self.settings = get_settings()

    async def _launch_browser(self):
        """Try bundled Chromium, then system Chrome, then Edge."""
        base_args: dict = {"headless": self.settings.playwright_headless}
        if self.settings.playwright_slow_mo_ms:
            base_args["slow_mo"] = self.settings.playwright_slow_mo_ms

        strategies = [
            ("bundled", dict(base_args)),
            ("chrome", {**base_args, "channel": "chrome"}),
            ("msedge", {**base_args, "channel": "msedge"}),
        ]

        last_error: Exception | None = None
        for mode, kwargs in strategies:
            try:
                browser = await self._playwright.chromium.launch(**kwargs)
                self._launch_mode = mode
                logger.info("browser_launched", mode=mode, headless=kwargs.get("headless"))
                return browser
            except Exception as exc:
                last_error = exc
                logger.debug("browser_launch_failed", mode=mode, error=str(exc))

        raise PlaywrightNotInstalledError(f"{install_hint()}\n\nDetail: {last_error}")

    async def start(self) -> None:
        if self._browser is not None:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._launch_browser()
        self._semaphore = asyncio.Semaphore(self.max_contexts)
        logger.info(
            "browser_pool_started",
            headless=self.settings.playwright_headless,
            mode=self._launch_mode,
        )

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("browser_pool_stopped")

    @asynccontextmanager
    async def acquire_page(
        self,
        locale: str = "vi-VN",
    ) -> AsyncIterator[Page]:
        await self.start()
        assert self._browser and self._semaphore

        async with self._semaphore:
            context_args: dict = {
                "user_agent": self.settings.user_agent,
                "locale": locale,
                "viewport": {"width": 1280, "height": 900},
            }
            if self.settings.proxy_url:
                context_args["proxy"] = {"server": self.settings.proxy_url}

            context: BrowserContext = await self._browser.new_context(**context_args)
            page = await context.new_page()
            page.set_default_timeout(int(self.settings.request_timeout_sec * 1000))
            try:
                yield page
            finally:
                await context.close()

    async def infinite_scroll(
        self,
        page: Page,
        scroll_selector: Optional[str] = None,
        max_scrolls: int = 20,
        wait_ms: int = 1500,
    ) -> int:
        """Scroll until no new content or max_scrolls reached."""
        prev_height = 0
        scrolls = 0

        for _ in range(max_scrolls):
            if scroll_selector:
                await page.locator(scroll_selector).last.scroll_into_view_if_needed()
            else:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            await page.wait_for_timeout(wait_ms)
            height = await page.evaluate("document.body.scrollHeight")

            if height == prev_height:
                break
            prev_height = height
            scrolls += 1

        return scrolls
