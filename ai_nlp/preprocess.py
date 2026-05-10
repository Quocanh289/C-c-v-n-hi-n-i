import re
import unicodedata
from typing import List, Tuple

def normalize_text(text: str) -> str:
    """Chuẩn hóa text: loại bỏ dấu cách, chuyển thường, v.v."""
    # Chuyển về thường
    text = text.lower()
    
    # Loại bỏ khoảng trắng dư thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Loại bỏ dấu và ký tự đặc biệt (giữ lại từ tiếng Việt)
    text = unicodedata.normalize('NFC', text)
    
    return text

def tokenize(text: str) -> List[str]:
    """Tách text thành các từ đơn."""
    text = normalize_text(text)
    # Tách theo dấu cách và dấu câu
    tokens = re.findall(r'\b\w+\b', text, re.UNICODE)
    return tokens

def remove_stopwords(tokens: List[str], stopwords: List[str]) -> List[str]:
    """Loại bỏ các từ phổ biến không có ý nghĩa."""
    return [token for token in tokens if token not in stopwords]

def extract_sentences(text: str) -> List[str]:
    """Tách text thành các câu."""
    # Tách theo dấu chấm, hỏi, cảm
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# Stopwords tiếng Việt cơ bản
VIETNAMESE_STOPWORDS = {
    'và', 'hay', 'hoặc', 'nhưng', 'vì', 'nếu', 'thì', 'là', 'cái', 'chiếc',
    'những', 'cái', 'trong', 'ngoài', 'trên', 'dưới', 'ở', 'với', 'từ', 'đến',
    'tại', 'cho', 'bởi', 'không', 'có', 'được', 'làm', 'đã', 'đang', 'sẽ',
    'của', 'để', 'như', 'cũng', 'mà', 'do', 'nên', 'rồi', 'khi', 'này',
    'kia', 'gì', 'ai', 'nào', 'đó', 'bao', 'lắm', 'rất', 'quá', 'chỉ',
    'tôi', 'tôi', 'tớ', 'mình', 'ta', 'bạn', 'anh', 'chị', 'em', 'cậu',
}

# Stopwords tiếng Anh cơ bản
ENGLISH_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'it', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which',
}

ALL_STOPWORDS = VIETNAMESE_STOPWORDS | ENGLISH_STOPWORDS

def detect_language(text: str) -> str:
    """Phát hiện ngôn ngữ (Vietnamese hoặc English)."""
    vietnamese_pattern = r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưứừửữựỳýỷỹỵđ]'
    vietnamese_count = len(re.findall(vietnamese_pattern, text.lower()))
    
    if vietnamese_count > len(text) * 0.1:  # >10% ký tự tiếng Việt
        return 'vietnamese'
    return 'english'
        "tokens": tokens
    }