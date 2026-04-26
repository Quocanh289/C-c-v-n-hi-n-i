# Sentiment classification module
def classify_sentiment(tokens: List[str]) -> str:
    # Placeholder logic
    negative_words = ["mệt", "buồn", "không muốn", "vô dụng"]
    if any(word in tokens for word in negative_words):
        return "negative"
    return "neutral"