from typing import Dict, List, Any
from preprocess import normalize_text, extract_sentences, detect_language
from sentiment_model import analyze_sentiment, get_sentiment_score
from emotion_model import get_primary_emotions
from symptom_extractor import extract_symptoms, get_symptom_score, format_symptom_display
from condition_mapper import map_symptoms_to_conditions, get_top_conditions
from risk_assessor import assess_risk_level, get_risk_signals, generate_recommendation

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Phân tích text đầu vào và trả kết quả chi tiết.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Dictionary with complete analysis results
    """
    # 1. Validate input
    if not text or len(text.strip()) < 10:
        return {
            'sentiment': 'unknown',
            'emotions': [],
            'severity': 'low',
            'risk_signals': [],
            'possible_related_conditions': [],
            'recommendation': 'Vui lòng cung cấp thêm thông tin để có phân tích chi tiết.'
        }
    
    # 2. Preprocess text
    normalized_text = normalize_text(text)
    language = detect_language(text)
    
    # 3. Sentiment analysis
    sentiment, sentiment_score = analyze_sentiment(text)
    
    # 4. Emotion extraction
    primary_emotions = get_primary_emotions(text, top_n=3)
    
    # 5. Symptom extraction
    symptom_scores = get_symptom_score(text)
    symptoms = extract_symptoms(text)
    
    # 6. Condition mapping
    conditions = map_symptoms_to_conditions(symptom_scores)
    top_conditions = get_top_conditions(conditions, top_n=3)
    
    # 7. Risk assessment
    risk_level = assess_risk_level(text, sentiment_score, symptom_scores, primary_emotions)
    risk_signals = get_risk_signals(text, symptom_scores, primary_emotions)
    
    # 8. Generate recommendation
    recommendation = generate_recommendation(risk_level, primary_emotions)
    
    # 9. Format response
    response = {
        'sentiment': sentiment,
        'emotions': primary_emotions,
        'severity': risk_level,
        'risk_signals': risk_signals,
        'possible_related_conditions': format_conditions(top_conditions),
        'recommendation': recommendation,
        'debug_info': {
            'language': language,
            'sentiment_score': round(sentiment_score, 3),
            'symptoms_detected': symptoms,
            'symptom_scores': {k: round(v, 3) for k, v in symptom_scores.items()},
        }
    }
    
    return response

def format_conditions(conditions: List[Dict]) -> List[Dict]:
    """
    Format conditions output for API response.
    
    Args:
        conditions: Raw conditions from mapper
    
    Returns:
        Formatted conditions for frontend
    """
    formatted = []
    for cond in conditions:
        formatted.append({
            'name': cond['name_vi'],
            'name_en': cond['name'],
            'confidence': round(cond['confidence'] * 100, 1),
            'reason': cond['reason'],
            'description': cond['description'],
        })
    return formatted

# Test function
if __name__ == '__main__':
    test_texts = [
        "Dạo này tôi không muốn gặp ai, mất ngủ, thấy mình vô dụng và không còn hứng thú với mọi thứ.",
        "I feel so happy and excited about life today!",
        "Tôi muốn chết, không thể chịu đựng được nữa.",
    ]
    
    for test_text in test_texts:
        print(f"\n{'='*60}")
        print(f"Input: {test_text}")
        print(f"{'='*60}")
        result = analyze_text(test_text)
        print(f"Sentiment: {result['sentiment']}")
        print(f"Emotions: {result['emotions']}")
        print(f"Severity: {result['severity']}")
        print(f"Risk Signals: {result['risk_signals']}")
        print(f"Conditions: {result['possible_related_conditions']}")
        print(f"Recommendation: {result['recommendation']}")