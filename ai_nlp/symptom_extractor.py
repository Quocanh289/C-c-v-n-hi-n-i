# Symptom extraction module
def extract_symptoms(tokens: List[str]) -> List[str]:
    # Placeholder logic
    symptoms = []
    symptom_keywords = {
        "mất ngủ": ["mất ngủ", "không ngủ"],
        "mất hứng thú": ["mất hứng thú", "không muốn"],
        "cô lập xã hội": ["không muốn gặp ai", "cô lập"]
    }
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in " ".join(tokens) for keyword in keywords):
            symptoms.append(symptom)
    return symptoms