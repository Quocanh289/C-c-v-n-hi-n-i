# Risk assessment module
def assess_risk(sentiment: str, emotions: List[str], symptoms: List[str]) -> str:
    # Placeholder logic
    if "không muốn sống" in " ".join(symptoms) or sentiment == "negative" and len(symptoms) > 2:
        return "high"
    return "medium"