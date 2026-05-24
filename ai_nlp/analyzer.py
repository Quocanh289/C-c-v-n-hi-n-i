# Main analyzer module
from typing import Dict

from .preprocess import preprocess_text
from .sentiment_model import classify_sentiment
from .emotion_model import classify_emotions
from .symptom_extractor import extract_symptoms
from .condition_mapper import map_to_conditions
from .risk_assessor import assess_risk


def analyze_text(text: str) -> Dict:
    processed = preprocess_text(text)
    sentiment = classify_sentiment(processed["tokens"])
    emotions = classify_emotions(processed["tokens"])
    symptoms = extract_symptoms(processed["tokens"])
    conditions = map_to_conditions(symptoms)
    severity = assess_risk(sentiment, emotions, symptoms)
    
    return {
        "sentiment": sentiment,
        "emotions": emotions,
        "severity": severity,
        "risk_signals": symptoms,
        "possible_related_conditions": conditions,
        "recommendation": "Đây không phải chẩn đoán y khoa. Hãy tham khảo chuyên gia."
    }