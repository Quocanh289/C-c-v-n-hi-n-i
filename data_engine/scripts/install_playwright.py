#!/usr/bin/env python3
"""Download Playwright Chromium for browser-based crawlers."""

import subprocess
import sys


def main():
    print("Installing Playwright Chromium (required for Facebook, TikTok, Threads)...")
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    print("Done. Verify with:")
    print(f"  {sys.executable} data_engine/scripts/run_crawl.py --platform facebook --target VietNam.official --limit 10")


if __name__ == "__main__":
    main()
