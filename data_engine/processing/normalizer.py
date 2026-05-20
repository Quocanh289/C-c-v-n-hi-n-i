"""
Text normalization for NLP dataset — preserves emotional signals.

DO NOT over-clean:
- Keep emojis, repeated chars, slang, expressive punctuation
"""

import re
import unicodedata
from typing import Optional

import ftfy

# Emoji pattern (broad Unicode ranges)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
MENTION_PATTERN = re.compile(r"@\w+")
# Only strip standalone hashtag markers, keep text after #
HASHTAG_CLEAN = re.compile(r"(?<=\s)#(?=\w)")

# Vietnamese diacritics + ASCII
VI_CHARS = set(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
)


def unicode_normalize(text: str) -> str:
    """Fix mojibake and normalize to NFC — preserves Vietnamese diacritics."""
    text = ftfy.fix_text(text)
    return unicodedata.normalize("NFC", text)


def extract_emojis(text: str) -> list[str]:
    return EMOJI_PATTERN.findall(text)


def remove_urls(text: str, replace: str = "") -> str:
    return URL_PATTERN.sub(replace, text)


def remove_mentions(text: str) -> str:
    return MENTION_PATTERN.sub("", text)


def normalize_repeated_chars(text: str, max_repeat: int = 4) -> str:
    """
    Collapse extreme repetition (aaaaaaa -> aaaa) but keep expressive repeats.
    e.g. "đỉnhhhh" -> "đỉnhhh", "soooo" -> "sooo"
    """
    pattern = re.compile(r"(.)\1{" + str(max_repeat) + r",}", re.UNICODE)
    return pattern.sub(lambda m: m.group(1) * max_repeat, text)


def light_clean(
    text: str,
    remove_url: bool = True,
    remove_mention: bool = True,
    normalize_repeat: bool = True,
) -> str:
    """
    Light cleaning pipeline — training text retains emotional markers.
    """
    text = unicode_normalize(text.strip())
    if remove_url:
        text = remove_urls(text)
    if remove_mention:
        text = remove_mentions(text)
    text = HASHTAG_CLEAN.sub("", text)
    # Collapse only excessive whitespace, not newlines meaning
    text = re.sub(r"[ \t]+", " ", text)
    if normalize_repeat:
        text = normalize_repeated_chars(text)
    return text.strip()


def tokenize_preserve(text: str) -> list[str]:
    """Simple tokenizer that keeps slang tokens and emojis separate."""
    tokens = []
    pos = 0
    for match in EMOJI_PATTERN.finditer(text):
        before = text[pos : match.start()].strip()
        if before:
            tokens.extend(before.split())
        tokens.append(match.group())
        pos = match.end()
    rest = text[pos:].strip()
    if rest:
        tokens.extend(rest.split())
    return [t for t in tokens if t]
