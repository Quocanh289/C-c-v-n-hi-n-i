"""Spam / bot detection — conservative to avoid dropping slang-heavy posts."""

import re
from typing import Optional

SPAM_PATTERNS = [
    re.compile(r"(?i)(click here|buy now|free money|earn \$)"),
    re.compile(r"(?i)(dm for|telegram|whatsapp).{0,20}(link|join)"),
    re.compile(r"http[s]?://\S+.*http[s]?://\S+"),  # multiple URLs
]
BOT_PATTERNS = [
    re.compile(r"^(.)\1{10,}$"),  # single char repeated
    re.compile(r"(?i)^(follow|subscribe|like).{0,20}(back|4follow)"),
]


def is_spam(
    text: str,
    min_unique_ratio: float = 0.3,
) -> tuple[bool, Optional[str]]:
    """
    Returns (is_spam, reason).
    Conservative — meme text with repetition is NOT spam.
    """
    if len(text) < 3:
        return True, "too_short"

    for pattern in SPAM_PATTERNS:
        if pattern.search(text):
            return True, "spam_pattern"

    for pattern in BOT_PATTERNS:
        if pattern.match(text):
            return True, "bot_pattern"

    tokens = text.split()
    if len(tokens) >= 5:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < min_unique_ratio and len(text) > 200:
            return True, "low_unique_ratio"

    return False, None
