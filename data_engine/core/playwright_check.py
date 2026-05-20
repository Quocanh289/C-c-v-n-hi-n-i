"""Playwright browser availability checks and install hints."""

from __future__ import annotations

import sys


def install_hint() -> str:
    py = sys.executable
    return (
        "Playwright browsers are not installed.\n\n"
        f"  {py} -m playwright install chromium\n\n"
        "Or install all browsers:\n\n"
        f"  {py} -m playwright install\n\n"
        "Facebook, TikTok, and Threads crawlers require this step."
    )


class PlaywrightNotInstalledError(RuntimeError):
    """Raised when no Chromium/Chrome/Edge browser can be launched."""


async def _try_launch(p, name: str, kwargs: dict) -> str:
    browser = await p.chromium.launch(**kwargs)
    await browser.close()
    return name


async def verify_playwright_async() -> str:
    """
    Try launching a browser with the async API.
    Returns which method worked: bundled | chrome | msedge.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise PlaywrightNotInstalledError(
            "playwright package not installed. Run: pip install playwright"
        ) from exc

    strategies = [
        ("bundled", {"headless": True}),
        ("chrome", {"channel": "chrome", "headless": True}),
        ("msedge", {"channel": "msedge", "headless": True}),
    ]

    last_error: Exception | None = None
    async with async_playwright() as p:
        for name, kwargs in strategies:
            try:
                return await _try_launch(p, name, kwargs)
            except Exception as exc:
                last_error = exc
                continue

    raise PlaywrightNotInstalledError(f"{install_hint()}\n\nDetail: {last_error}")


def verify_playwright_sync() -> str:
    """
    Sync check — use only outside an asyncio loop (e.g. install script).
    From async code, use verify_playwright_async() instead.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PlaywrightNotInstalledError(
            "playwright package not installed. Run: pip install playwright"
        ) from exc

    strategies = [
        ("bundled", {"headless": True}),
        ("chrome", {"channel": "chrome", "headless": True}),
        ("msedge", {"channel": "msedge", "headless": True}),
    ]

    last_error: Exception | None = None
    with sync_playwright() as p:
        for name, kwargs in strategies:
            try:
                browser = p.chromium.launch(**kwargs)
                browser.close()
                return name
            except Exception as exc:
                last_error = exc
                continue

    raise PlaywrightNotInstalledError(f"{install_hint()}\n\nDetail: {last_error}")
