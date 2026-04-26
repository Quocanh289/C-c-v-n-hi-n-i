# Emotion classification module
def classify_emotions(tokens: List[str]) -> List[str]:
    # Placeholder logic
    emotions = []
    if "buồn" in tokens:
        emotions.append("sadness")
    if "mệt" in tokens:
        emotions.append("fatigue")
    return emotions