# Risk assessment module
def assess_risk(sentiment: str, emotions: list[str], symptoms: list[str]) -> str:
    # Placeholder logic
    if "không muốn sống" in " ".join(symptoms) or sentiment == "negative" and len(symptoms) > 2:
        return "high"
    return "medium"