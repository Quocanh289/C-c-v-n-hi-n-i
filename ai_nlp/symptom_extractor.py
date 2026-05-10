from typing import List, Dict
import re

# Mental health symptoms mapping
MENTAL_HEALTH_SYMPTOMS = {
    'insomnia': {
        'keywords_vi': ['mất ngủ', 'không ngủ được', 'khó ngủ', 'thức đêm', 'mất vài giờ ngủ'],
        'keywords_en': ['insomnia', 'sleep deprivation', 'cannot sleep', 'sleepless', 'cannot fall asleep'],
    },
    'loss_of_interest': {
        'keywords_vi': ['mất hứng thú', 'không có hứng thú', 'không muốn làm gì', 'chán làm gì'],
        'keywords_en': ['loss of interest', 'no interest', 'anhedonia', 'no motivation', 'apathy'],
    },
    'fatigue': {
        'keywords_vi': ['mệt mỏi', 'mệt', 'uể oải', 'kiệt sức', 'không có sức'],
        'keywords_en': ['tired', 'fatigue', 'exhausted', 'weary', 'no energy'],
    },
    'negative_self_talk': {
        'keywords_vi': ['tự đánh giá thấp', 'tự xúc phạm', 'tôi vô dụng', 'tôi không xứng đáng', 'tôi tệ'],
        'keywords_en': ['worthless', 'useless', 'hate myself', 'no good', 'failure'],
    },
    'social_isolation': {
        'keywords_vi': ['không muốn gặp ai', 'tránh mọi người', 'cô lập', 'rút vào', 'không muốn nói chuyện'],
        'keywords_en': ['avoid people', 'isolation', 'withdrawn', 'no one understands', 'alone'],
    },
    'concentration_issues': {
        'keywords_vi': ['không tập trung', 'sao nhãng', 'lú lẫn', 'tâm trí rối loạn', 'không thể tập trung'],
        'keywords_en': ['cannot concentrate', 'cannot focus', 'scattered thoughts', 'foggy mind', 'absent-minded'],
    },
    'appetite_changes': {
        'keywords_vi': ['mất cảm giác đói', 'ăn quá nhiều', 'không có cảm giác', 'không muốn ăn'],
        'keywords_en': ['loss of appetite', 'overeating', 'appetite changes', 'eating too much'],
    },
    'suicidal_thoughts': {
        'keywords_vi': ['muốn chết', 'tự tử', 'sống không có ý nghĩa', 'tự làm hại bản thân'],
        'keywords_en': ['suicide', 'suicidal', 'want to die', 'self-harm', 'want to hurt myself'],
    },
    'physical_pain': {
        'keywords_vi': ['đau đầu', 'đau lưng', 'đau ngực', 'không khỏe', 'bị bệnh'],
        'keywords_en': ['headache', 'body pain', 'chest pain', 'physical pain', 'illness'],
    },
    'irritability': {
        'keywords_vi': ['dễ cáu', 'dễ nổi giận', 'nóng tính', 'không kiên nhẫn', 'dễ chóng mặt'],
        'keywords_en': ['irritable', 'short-tempered', 'moody', 'grumpy', 'easily annoyed'],
    },
    'guilt': {
        'keywords_vi': ['cảm thấy có lỗi', 'tội lỗi', 'hối hận', 'xấu hổ', 'tự trách'],
        'keywords_en': ['guilt', 'guilty', 'regret', 'shame', 'self-blame'],
    },
    'paranoia': {
        'keywords_vi': ['mọi người đều chống lại tôi', 'ai cũng muốn hại tôi', 'không tin ai'],
        'keywords_en': ['paranoid', 'everyone is against me', 'cannot trust anyone', 'suspicious'],
    }
}

def extract_symptoms(text: str) -> List[str]:
    """
    Trích xuất danh sách triệu chứng tâm thần từ text.
    
    Returns:
        List of detected symptoms
    """
    text_lower = text.lower()
    detected_symptoms = []
    
    for symptom, keywords_dict in MENTAL_HEALTH_SYMPTOMS.items():
        keywords_vi = keywords_dict.get('keywords_vi', [])
        keywords_en = keywords_dict.get('keywords_en', [])
        all_keywords = keywords_vi + keywords_en
        
        for keyword in all_keywords:
            if keyword.lower() in text_lower:
                if symptom not in detected_symptoms:
                    detected_symptoms.append(symptom)
                break
    
    return detected_symptoms

def get_symptom_score(text: str) -> Dict[str, float]:
    """
    Tính điểm cho mỗi triệu chứng (0-1).
    
    Returns:
        Dict mapping symptom -> score
    """
    text_lower = text.lower()
    scores = {}
    
    for symptom, keywords_dict in MENTAL_HEALTH_SYMPTOMS.items():
        keywords_vi = keywords_dict.get('keywords_vi', [])
        keywords_en = keywords_dict.get('keywords_en', [])
        all_keywords = keywords_vi + keywords_en
        
        # Đếm số lần xuất hiện của keywords
        count = sum(text_lower.count(keyword.lower()) for keyword in all_keywords)
        
        # Normalize score (0-1)
        score = min(count / 3, 1.0)  # Max at 3 occurrences
        
        if score > 0:
            scores[symptom] = score
    
    return scores

def get_risk_symptoms(text: str) -> List[str]:
    """
    Trích xuất các triệu chứng nguy hiểm (tự tử, tự làm hại).
    
    Returns:
        List of dangerous symptoms
    """
    text_lower = text.lower()
    dangerous = ['suicidal_thoughts']
    
    dangerous_symptoms = []
    for symptom in dangerous:
        if symptom in MENTAL_HEALTH_SYMPTOMS:
            keywords = (MENTAL_HEALTH_SYMPTOMS[symptom].get('keywords_vi', []) + 
                       MENTAL_HEALTH_SYMPTOMS[symptom].get('keywords_en', []))
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    dangerous_symptoms.append(symptom)
                    break
    
    return dangerous_symptoms

def format_symptom_display(symptom: str) -> str:
    """Chuyển đổi symptom key thành hiển thị dễ đọc."""
    display_map = {
        'insomnia': 'Mất ngủ',
        'loss_of_interest': 'Mất hứng thú',
        'fatigue': 'Mệt mỏi',
        'negative_self_talk': 'Tự đánh giá tiêu cực',
        'social_isolation': 'Cô lập xã hội',
        'concentration_issues': 'Không tập trung',
        'appetite_changes': 'Thay đổi cơn đói',
        'suicidal_thoughts': 'Ý nghĩ tự tử',
        'physical_pain': 'Đau thể chất',
        'irritability': 'Dễ cáu kỉnh',
        'guilt': 'Cảm giác tội lỗi',
        'paranoia': 'Hoang tưởng',
    }
    return display_map.get(symptom, symptom)