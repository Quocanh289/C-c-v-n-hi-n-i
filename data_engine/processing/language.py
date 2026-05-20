"""Language detection for Vietnamese / English / mixed code-switching."""

import re
from typing import Literal

from data_engine.processing.normalizer import VI_CHARS

LanguageLabel = Literal["vi", "en", "mixed"]


def detect_language(text: str) -> LanguageLabel:
    """
    Heuristic language detection optimized for Vi-En social media.
    Falls back to langdetect when available.
    """
    try:
        import langdetect
        from langdetect import DetectorFactory

        DetectorFactory.seed = 0
        detected = langdetect.detect(text)
        if detected == "vi":
            return "vi"
        if detected == "en":
            return "en"
    except Exception:
        pass

    vi_count = sum(1 for c in text if c in VI_CHARS)
    ascii_words = re.findall(r"[a-zA-Z]{2,}", text)
    vi_words = re.findall(
        r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]+",
        text,
    )

    has_vi = vi_count > 2 or len(vi_words) > 0
    has_en = len(ascii_words) > 0

    if has_vi and has_en:
        return "mixed"
    if has_vi:
        return "vi"
    if has_en:
        return "en"
    return "mixed"
