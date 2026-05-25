"""Translate API for auto-translating Vietnamese text to English."""
import logging
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/translate", tags=["translate"])


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    source_lang: str = Field(default="vi")
    target_lang: str = Field(default="en")


class TranslateResponse(BaseModel):
    translated_text: str
    source_lang: str = "vi"
    target_lang: str = "en"
    method: str = "dictionary"


# Simple Vi→En dictionary for common mental health phrases
VI_EN_MAP = {
    "tôi buồn": "I feel sad",
    "tôi rất buồn": "I am very sad",
    "buồn quá": "I am so sad",
    "chán nản": "I feel hopeless",
    "mệt mỏi": "I am tired and exhausted",
    "áp lực": "I am under pressure",
    "căng thẳng": "I am stressed",
    "lo lắng": "I am worried and anxious",
    "rất lo lắng": "I am very anxious",
    "sợ hãi": "I am scared and fearful",
    "hoảng sợ": "I am panicking",
    "tuyệt vọng": "I feel hopeless and desperate",
    "vô dụng": "I feel worthless",
    "vô vọng": "I feel hopeless",
    "không muốn sống": "I don't want to live anymore",
    "muốn chết": "I want to die",
    "tự tử": "I want to commit suicide",
    "đau khổ": "I am suffering and in pain",
    "cô đơn": "I feel lonely and alone",
    "mất ngủ": "I have insomnia, can't sleep",
    "không ngủ được": "I cannot sleep at night",
    "mất hứng thú": "I have lost interest in everything",
    "không còn hứng thú": "I am no longer interested in anything",
    "không tập trung": "I cannot focus or concentrate",
    "bồn chồn": "I feel restless and anxious",
    "khó thở": "I have difficulty breathing",
    "tim đập nhanh": "My heart is racing fast",
    "hoảng loạn": "I am having a panic attack",
    "dao động cảm xúc": "I have extreme mood swings",
    "vui": "I feel happy",
    "hạnh phúc": "I am happy and joyful",
    "tuyệt vời": "I feel amazing and wonderful",
    "yêu đời": "I love life",
    "bình thường": "I feel normal and fine",
    "không sao": "I am okay, nothing is wrong",
    "khỏe": "I am healthy and fine",
}

VIETNAMESE_CHAR_RE = re.compile(
    r"[\u0103\u00e2\u0111\u00ea\u00f4\u01a1\u01b0"
    r"\u00e0\u00e1\u1ea3\u00e3\u1ea1\u1eb1\u1eaf\u1eb3\u1eb5\u1eb7"
    r"\u1ea7\u1ea5\u1ea9\u1eab\u1ead\u00e8\u00e9\u1ebb\u1ebd\u1eb9"
    r"\u1ec1\u1ebf\u1ec3\u1ec5\u1ec7\u00ec\u00ed\u1ec9\u0129\u1ecb"
    r"\u00f2\u00f3\u1ecf\u00f5\u1ecd\u1ed3\u1ed1\u1ed5\u1ed7\u1ed9"
    r"\u1edd\u1edb\u1edf\u1ee1\u1ee3\u00f9\u00fa\u1ee7\u0169\u1ee5"
    r"\u1eeb\u1ee9\u1eed\u1eef\u1ef1\u1ef3\u00fd\u1ef7\u1ef9\u1ef5]",
    re.IGNORECASE,
)


def contains_vietnamese(text: str) -> bool:
    """Return True when text contains Vietnamese-specific characters."""
    return bool(VIETNAMESE_CHAR_RE.search(text))


def translate_simple(text: str) -> str:
    """Dictionary-based Vi→En translation."""
    lowered = text.lower().strip()
    # Check full phrase matches first
    for vi, en in sorted(VI_EN_MAP.items(), key=lambda x: -len(x[0])):
        if vi in lowered:
            return en
    # Simple word mapping fallback
    word_map = {
        "tôi": "I", "bạn": "you", "nó": "it", "chúng": "we",
        "và": "and", "nhưng": "but", "của": "of",
        "là": "is", "có": "have", "không": "not", "rất": "very",
        "quá": "too", "đang": "am", "sẽ": "will",
        "đã": "have", "em": "I", "anh": "I",
        "thấy": "feel", "cảm thấy": "feel",
        "ngày": "day", "hôm nay": "today",
        "mọi": "every", "thứ": "thing", "người": "people",
        "thời gian": "time", "cuộc sống": "life",
        "việc": "work", "học": "study",
        "làm": "do", "nghĩ": "think",
        "biết": "know", "hiểu": "understand",
        "muốn": "want", "cần": "need",
        "đi": "go", "đến": "come", "ở": "at",
        "với": "with", "cho": "for", "từ": "from",
        "xin": "please", "cảm ơn": "thank you",
    }
    words = lowered.split()
    translated = []
    for w in words:
        translated.append(word_map.get(w, w))
    return " ".join(translated)


@router.post("", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    try:
        result = translate_simple(request.text)
        return TranslateResponse(
            translated_text=result,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            method="dictionary",
        )
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
