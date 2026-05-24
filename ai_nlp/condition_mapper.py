# Condition mapping module
from typing import Dict, List


def map_to_conditions(symptoms: List[str]) -> List[Dict]:
    # Placeholder logic
    conditions: List[Dict] = []
    if "mất hứng thú" in symptoms:
        conditions.append({
            "name": "Trầm cảm",
            "confidence": 0.7,
            "reason": "Mất hứng thú là dấu hiệu của trầm cảm"
        })
    return conditions