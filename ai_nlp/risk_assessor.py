from typing import Dict, List, Tuple
import re

RISK_KEYWORDS = {
    'critical': {
        'vi': ['tự tử', 'muốn chết', 'sống không có ý nghĩa', 'tự làm hại', 'bắt đầu'],
        'en': ['suicide', 'kill myself', 'want to die', 'self-harm', 'hurt myself'],
    },
    'high': {
        'vi': ['tuyệt vọng', 'không còn hy vọng', 'mất mọi thứ', 'không thể chịu đựng'],
        'en': ['hopeless', 'desperate', 'cannot bear', 'unbearable', 'cannot take it'],
    },
    'medium': {
        'vi': ['mất hứng thú', 'không muốn sống', 'chán đời', 'mệt mỏi'],
        'en': ['no interest in life', 'tired of living', 'fatigue', 'exhausted'],
    }
}

def assess_risk_level(
    text: str,
    sentiment_score: float,
    symptom_scores: Dict[str, float],
    primary_emotions: List[str]
) -> str:
    """
    Đánh giá mức độ rủi ro của người dùng.
    
    Args:
        text: Input text
        sentiment_score: Sentiment score (0-1)
        symptom_scores: Dictionary of symptom scores
        primary_emotions: List of primary emotions
    
    Returns:
        Risk level: 'critical', 'high', 'medium', 'low'
    """
    risk_score = 0.0
    
    # 1. Kiểm tra từ khóa rủi ro (critical)
    text_lower = text.lower()
    critical_keywords = RISK_KEYWORDS['critical']['vi'] + RISK_KEYWORDS['critical']['en']
    for keyword in critical_keywords:
        if keyword.lower() in text_lower:
            return 'critical'
    
    # 2. Sentiment score (weighted)
    if sentiment_score < 0.3:
        risk_score += 0.4
    elif sentiment_score < 0.5:
        risk_score += 0.2
    
    # 3. Symptom analysis (weighted)
    dangerous_symptoms = ['suicidal_thoughts', 'negative_self_talk', 'hopelessness']
    dangerous_score = sum(symptom_scores.get(s, 0) for s in dangerous_symptoms)
    if dangerous_score > 0:
        risk_score += min(dangerous_score * 0.35, 0.35)
    
    # 4. Emotion analysis (weighted)
    if 'hopelessness' in primary_emotions or 'sadness' in primary_emotions:
        risk_score += 0.2
    
    # 5. Multiple negative indicators (weighted)
    if len(primary_emotions) >= 2 and sentiment_score < 0.4:
        risk_score += 0.1
    
    # 6. High keyword frequency
    high_keywords = RISK_KEYWORDS['high']['vi'] + RISK_KEYWORDS['high']['en']
    high_keyword_count = sum(text_lower.count(kw.lower()) for kw in high_keywords)
    if high_keyword_count >= 2:
        risk_score += 0.15
    
    # Determine risk level
    if risk_score >= 0.7:
        return 'critical'
    elif risk_score >= 0.5:
        return 'high'
    elif risk_score >= 0.3:
        return 'medium'
    else:
        return 'low'

def get_risk_signals(
    text: str,
    symptom_scores: Dict[str, float],
    primary_emotions: List[str]
) -> List[str]:
    """
    Trích xuất các tín hiệu rủi ro từ text.
    
    Returns:
        List of risk signals detected
    """
    risk_signals = []
    text_lower = text.lower()
    
    # Mapping from symptom keys to human-readable Vietnamese text
    symptom_text_map = {
        'insomnia': 'mất ngủ',
        'loss_of_interest': 'mất hứng thú',
        'fatigue': 'mệt mỏi',
        'negative_self_talk': 'tự đánh giá bản thân tiêu cực',
        'social_isolation': 'cô lập xã hội',
        'concentration_issues': 'khó tập trung',
        'appetite_changes': 'thay đổi cơn đói',
        'suicidal_thoughts': 'ý nghĩ tự tử',
        'physical_pain': 'đau thể chất',
        'irritability': 'dễ cáu kỉnh',
        'guilt': 'cảm giác tội lỗi',
        'paranoia': 'hoang tưởng',
    }
    
    # Add symptoms as risk signals
    for symptom, score in symptom_scores.items():
        if score > 0:
            display = symptom_text_map.get(symptom, symptom)
            risk_signals.append(display)
    
    # Add emotion-based signals
    emotion_text_map = {
        'sadness': 'cảm xúc buồn bã',
        'anxiety': 'lo lắng, căng thẳng',
        'hopelessness': 'cảm giác vô vọng',
        'worthlessness': 'cảm giác không có giá trị',
    }
    
    for emotion in primary_emotions:
        if emotion in emotion_text_map:
            risk_signals.append(emotion_text_map[emotion])
    
    # Check for explicit dangerous statements
    dangerous_phrases_vi = [
        'muốn chết', 'tự tử', 'sống không có ý nghĩa',
        'không ai cần tôi', 'mọi người tốt hơn nếu tôi biến mất'
    ]
    dangerous_phrases_en = [
        'want to die', 'suicide', 'life is pointless',
        'no one needs me', 'better off without me'
    ]
    
    for phrase in dangerous_phrases_vi + dangerous_phrases_en:
        if phrase.lower() in text_lower:
            if 'tuyên bố tự tử' not in risk_signals:
                risk_signals.append('tuyên bố tự tử hoặc tự làm hại')
                break
    
    return risk_signals[:8]  # Limit to top 8 signals

def generate_recommendation(risk_level: str, primary_emotions: List[str]) -> str:
    """
    Tạo khuyến cáo dựa trên mức độ rủi ro.
    
    Returns:
        Vietnamese recommendation text
    """
    base_message = "Đây không phải chẩn đoán y khoa. "
    
    if risk_level == 'critical':
        return (base_message + 
                "⚠️ CẢNH BÁO: Các biểu hiện cho thấy mức độ rủi ro NGHIÊM TRỌNG. "
                "Nếu có ý nghĩ tự làm hại bản thân hoặc tự tử, vui lòng liên hệ ngay với: "
                "📞 Đường dây hỗ trợ khẩn cấp tâm thần hoặc đến bệnh viện gần nhất.")
    elif risk_level == 'high':
        return (base_message + 
                "⚠️ Tình trạng của bạn cho thấy mức độ rủi ro CAO. "
                "Nên tư vấn với chuyên gia tâm lý hoặc bác sĩ tâm thần sớm nhất.")
    elif risk_level == 'medium':
        return (base_message + 
                "Nên theo dõi tình trạng của bạn và xem xét tư vấn với chuyên gia tâm lý "
                "để có hỗ trợ và lời khuyên phù hợp.")
    else:
        return (base_message + 
                "Bạn có thể cân nhắc tìm hiểu thêm về sức khỏe tâm thần "
                "hoặc nói chuyện với người thân được tin tưởng.")