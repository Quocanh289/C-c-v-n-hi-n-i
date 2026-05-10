from typing import List, Dict
import re

# Emotion keywords mapping
EMOTION_KEYWORDS = {
    'sadness': {
        'vi': ['buồn', 'trist', 'khóc', 'mất tinh thần', 'lầm lũi', 'rủi sở', 'chán nản', 'uể oải'],
        'en': ['sad', 'crying', 'depressed', 'sorrowful', 'blue', 'gloomy', 'downhearted']
    },
    'anxiety': {
        'vi': ['lo lắng', 'lo sợ', 'căng thẳng', 'thất thần', 'bồn chồn', 'hồi hộp', 'trăn trở'],
        'en': ['anxious', 'worried', 'stressed', 'nervous', 'tense', 'fearful', 'panic']
    },
    'anger': {
        'vi': ['tức giận', 'nổi giận', 'phẫn nộ', 'tức', 'lên cơn', 'tối mặt', 'giận dữ'],
        'en': ['angry', 'furious', 'mad', 'enraged', 'irritated', 'irate', 'livid']
    },
    'hopelessness': {
        'vi': ['tuyệt vọng', 'vô vọng', 'không còn hy vọng', 'mất hy vọng', 'tuyệt vong'],
        'en': ['hopeless', 'desperate', 'helpless', 'despairing', 'defeated']
    },
    'loneliness': {
        'vi': ['cô đơn', 'cô lập', 'cô lẻ', 'một mình', 'không ai hiểu'],
        'en': ['lonely', 'isolated', 'alone', 'abandoned', 'isolated']
    },
    'worthlessness': {
        'vi': ['vô dụng', 'vô giá trị', 'tự đánh giá thấp', 'không xứng đáng', 'tệ hại'],
        'en': ['worthless', 'useless', 'valueless', 'unworthy', 'inadequate']
    },
    'joy': {
        'vi': ['vui', 'vui vẻ', 'hạnh phúc', 'phấn khởi', 'hân hoan', 'tươi cười'],
        'en': ['happy', 'joyful', 'delighted', 'cheerful', 'elated', 'thrilled']
    },
    'love': {
        'vi': ['yêu', 'yêu thích', 'thích', 'yêu mến', 'quý'],
        'en': ['love', 'adore', 'cherish', 'affection', 'devoted']
    },
    'confusion': {
        'vi': ['bối rối', 'lú lẫn', 'không hiểu', 'hối hỡn', 'mơ hồ'],
        'en': ['confused', 'bewildered', 'puzzled', 'perplexed', 'disoriented']
    },
    'shame': {
        'vi': ['xấu hổ', 'tự hổ', 'lấy làm xấu hổ', 'hổ thẹn', 'tủi nhục'],
        'en': ['ashamed', 'embarrassed', 'humiliated', 'disgraced', 'mortified']
    }
}

def extract_emotions(text: str) -> List[str]:
    """
    Trích xuất danh sách cảm xúc từ text.
    
    Returns:
        List of emotions detected (e.g., ['sadness', 'hopelessness', 'loneliness'])
    """
    text_lower = text.lower()
    detected_emotions = []
    
    for emotion, keywords_dict in EMOTION_KEYWORDS.items():
        all_keywords = keywords_dict.get('vi', []) + keywords_dict.get('en', [])
        
        # Kiểm tra nếu bất kỳ keyword của cảm xúc nào xuất hiện trong text
        for keyword in all_keywords:
            if keyword in text_lower:
                if emotion not in detected_emotions:
                    detected_emotions.append(emotion)
                break
    
    return detected_emotions if detected_emotions else ['neutral']

def get_emotion_intensity(text: str) -> Dict[str, float]:
    """
    Tính độ mạnh của mỗi cảm xúc (0-1).
    
    Returns:
        Dict mapping emotion -> intensity score
    """
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords_dict in EMOTION_KEYWORDS.items():
        all_keywords = keywords_dict.get('vi', []) + keywords_dict.get('en', [])
        
        # Đếm số lần xuất hiện
        count = sum(text_lower.count(keyword) for keyword in all_keywords)
        
        # Normalize score (0-1)
        intensity = min(count / 5, 1.0)  # Max at 5 occurrences
        
        if intensity > 0:
            emotion_scores[emotion] = intensity
    
    return emotion_scores

def get_primary_emotions(text: str, top_n: int = 3) -> List[str]:
    """
    Lấy cảm xúc chính (top N).
    
    Args:
        text: Input text
        top_n: Số cảm xúc chính cần lấy
    
    Returns:
        List of top N emotions by intensity
    """
    scores = get_emotion_intensity(text)
    
    if not scores:
        return ['neutral']
    
    # Sắp xếp theo intensity giảm dần
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [emotion for emotion, _ in sorted_emotions[:top_n]]