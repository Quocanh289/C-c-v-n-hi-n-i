from typing import List, Dict
import re

# Mental health conditions mapping based on symptoms
CONDITION_SYMPTOM_MAP = {
    'Depression': {
        'primary_symptoms': ['sadness', 'loss_of_interest', 'fatigue', 'negative_self_talk'],
        'secondary_symptoms': ['insomnia', 'appetite_changes', 'concentration_issues', 'guilt'],
        'name_vi': 'Trầm cảm',
        'description': 'Persistent mood disorder characterized by sadness, loss of interest, and hopelessness',
        'severity_threshold': 0.6,
    },
    'Anxiety Disorder': {
        'primary_symptoms': ['anxiety', 'concentration_issues', 'irritability'],
        'secondary_symptoms': ['insomnia', 'physical_pain', 'fatigue'],
        'name_vi': 'Rối loạn lo âu',
        'description': 'Excessive worry and anxiety affecting daily activities',
        'severity_threshold': 0.5,
    },
    'Social Anxiety': {
        'primary_symptoms': ['social_isolation', 'anxiety', 'negative_self_talk'],
        'secondary_symptoms': ['concentration_issues', 'insomnia'],
        'name_vi': 'Rối loạn lo âu xã hội',
        'description': 'Excessive fear in social situations and fear of judgment',
        'severity_threshold': 0.55,
    },
    'Insomnia Disorder': {
        'primary_symptoms': ['insomnia', 'fatigue', 'concentration_issues'],
        'secondary_symptoms': ['anxiety', 'irritability'],
        'name_vi': 'Rối loạn mất ngủ',
        'description': 'Chronic inability to fall or stay asleep',
        'severity_threshold': 0.4,
    },
    'Bipolar Disorder': {
        'primary_symptoms': ['extreme_mood_changes', 'irritability', 'concentration_issues'],
        'secondary_symptoms': ['insomnia', 'suicidal_thoughts'],
        'name_vi': 'Rối loạn lưỡng cực',
        'description': 'Alternating episodes of mania and depression',
        'severity_threshold': 0.65,
    },
    'PTSD': {
        'primary_symptoms': ['anxiety', 'social_isolation', 'irritability'],
        'secondary_symptoms': ['insomnia', 'concentration_issues', 'guilt'],
        'name_vi': 'Chứng rối loạn căng thẳng sau chấn thương',
        'description': 'Psychological disorder following traumatic events',
        'severity_threshold': 0.6,
    },
    'OCD': {
        'primary_symptoms': ['anxiety', 'concentration_issues', 'guilt'],
        'secondary_symptoms': ['insomnia', 'irritability'],
        'name_vi': 'Rối loạn ám ảnh-cưỡng chế',
        'description': 'Intrusive thoughts and repetitive behaviors',
        'severity_threshold': 0.55,
    },
}

def map_symptoms_to_conditions(symptom_scores: Dict[str, float]) -> List[Dict]:
    """
    Ánh xạ các triệu chứng sang các tình trạng tâm thần có thể liên quan.
    
    Args:
        symptom_scores: Dict mapping symptom -> score (0-1)
    
    Returns:
        List of conditions with confidence scores, sorted by confidence
    """
    condition_scores = {}
    
    for condition, config in CONDITION_SYMPTOM_MAP.items():
        primary_symptoms = config['primary_symptoms']
        secondary_symptoms = config['secondary_symptoms']
        
        # Tính điểm cho condition
        primary_score = sum(symptom_scores.get(s, 0) for s in primary_symptoms) / len(primary_symptoms)
        secondary_score = sum(symptom_scores.get(s, 0) for s in secondary_symptoms) / len(secondary_symptoms) if secondary_symptoms else 0
        
        # Primary symptoms có trọng số cao hơn
        final_score = (primary_score * 0.7) + (secondary_score * 0.3)
        
        if final_score >= config['severity_threshold']:
            condition_scores[condition] = {
                'confidence': min(final_score, 1.0),
                'name_vi': config['name_vi'],
                'description': config['description'],
            }
    
    # Sắp xếp theo confidence giảm dần
    sorted_conditions = sorted(
        condition_scores.items(),
        key=lambda x: x[1]['confidence'],
        reverse=True
    )
    
    return [
        {
            'name': condition,
            'name_vi': config['name_vi'],
            'confidence': config['confidence'],
            'description': config['description'],
            'reason': generate_reason(condition, symptom_scores)
        }
        for condition, config in sorted_conditions
    ]

def generate_reason(condition: str, symptom_scores: Dict[str, float]) -> str:
    """
    Tạo giải thích vì sao condition này được gợi ý.
    
    Args:
        condition: Condition name
        symptom_scores: Dict of detected symptoms and their scores
    
    Returns:
        Explanation text in Vietnamese
    """
    if condition not in CONDITION_SYMPTOM_MAP:
        return ""
    
    config = CONDITION_SYMPTOM_MAP[condition]
    primary_symptoms = config['primary_symptoms']
    
    # Format symptom names for display
    symptom_display_map = {
        'sadness': 'cảm giác buồn',
        'loss_of_interest': 'mất hứng thú',
        'fatigue': 'mệt mỏi',
        'negative_self_talk': 'tự đánh giá tiêu cực',
        'insomnia': 'mất ngủ',
        'appetite_changes': 'thay đổi cơn đói',
        'concentration_issues': 'khó tập trung',
        'anxiety': 'lo lắng',
        'irritability': 'dễ cáu kỉnh',
        'social_isolation': 'cô lập xã hội',
        'guilt': 'cảm giác tội lỗi',
        'suicidal_thoughts': 'ý nghĩ tự tử',
    }
    
    detected = []
    for symptom in primary_symptoms:
        if symptom in symptom_scores and symptom_scores[symptom] > 0:
            display = symptom_display_map.get(symptom, symptom)
            detected.append(display)
    
    if detected:
        return f"Có dấu hiệu {', '.join(detected[:2])}"
    return f"Phù hợp với các đặc điểm của {config['name_vi']}"

def get_top_conditions(conditions: List[Dict], top_n: int = 3) -> List[Dict]:
    """Lấy top N tình trạng có confidence cao nhất."""
    return conditions[:top_n]