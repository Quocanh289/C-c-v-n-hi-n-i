from typing import Tuple
import re

# Dictionary-based sentiment analysis (cơ bản)
# Trong production, sẽ dùng transformers như DistilBERT, PhoBERT

POSITIVE_WORDS_VI = {
    'vui', 'hạnh phúc', 'tuyệt vời', 'tốt', 'tuyệt', 'yêu', 'yêu thích',
    'thích', 'thích thú', 'phấn khởi', 'hạnh phúc', 'vui vẻ', 'mừng',
    'cảm ơn', 'tốt bụng', 'tử tế', 'thân thiện', 'lạc quan', 'hy vọng',
    'mong chờ', 'háo hức', 'hứng thú', 'say mê', 'yêu', 'quý'
}

NEGATIVE_WORDS_VI = {
    'buồn', 'trist', 'bất hạnh', 'xấu', 'tệ', 'ghét', 'ghét bỏ',
    'không thích', 'thất vọng', 'chán nản', 'chán', 'tuyệt vọng',
    'tuyệt vong', 'mất ngủ', 'căng thẳng', 'lo lắng', 'sợ', 'sợ hãi',
    'tức giận', 'tức', 'nổi giận', 'phẫn nộ', 'ghen tị', 'ghen',
    'cô đơn', 'cô lập', 'vô dụng', 'vô giá trị', 'tự tử', 'chết',
    'bệnh', 'bệnh tật', 'đau', 'khổ', 'thống khổ', 'thiệt hại',
}

POSITIVE_WORDS_EN = {
    'happy', 'joyful', 'wonderful', 'great', 'excellent', 'love', 'like',
    'enjoy', 'excited', 'delighted', 'grateful', 'kind', 'friendly',
    'optimistic', 'hope', 'hopeful', 'eager', 'interested', 'amazing',
    'beautiful', 'perfect', 'good', 'nice', 'blessed', 'grateful'
}

NEGATIVE_WORDS_EN = {
    'sad', 'unhappy', 'bad', 'terrible', 'awful', 'hate', 'dislike',
    'disappointed', 'depressed', 'anxious', 'worried', 'scared', 'afraid',
    'angry', 'furious', 'jealous', 'lonely', 'isolated', 'worthless',
    'useless', 'suicidal', 'sick', 'ill', 'pain', 'suffer', 'suffering',
    'devastated', 'broken', 'miserable', 'desperate', 'helpless'
}

def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Phân tích sentiment của text.
    
    Returns:
        (sentiment: 'positive', 'neutral', 'negative', score: 0.0-1.0)
    """
    text_lower = text.lower()
    
    # Đếm từ tích cực/tiêu cực
    positive_count = sum(1 for word in POSITIVE_WORDS_VI | POSITIVE_WORDS_EN 
                        if word in text_lower)
    negative_count = sum(1 for word in NEGATIVE_WORDS_VI | NEGATIVE_WORDS_EN 
                        if word in text_lower)
    
    # Tính điểm
    total = positive_count + negative_count
    if total == 0:
        return 'neutral', 0.5
    
    score = positive_count / total
    
    # Phân loại
    if score > 0.6:
        sentiment = 'positive'
    elif score < 0.4:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment, score

def get_sentiment_score(text: str) -> float:
    """Lấy điểm sentiment (0-1, 0.5 = neutral)."""
    _, score = analyze_sentiment(text)
    return score

def get_sentiment_label(text: str) -> str:
    """Lấy nhãn sentiment."""
    sentiment, _ = analyze_sentiment(text)
    return sentiment