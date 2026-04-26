# Text preprocessing module
import re
from typing import Dict, List

def preprocess_text(text: str) -> Dict:
    # Normalize to lowercase
    clean_text = text.lower()
    # Remove extra whitespace and emojis (basic)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    
    # Simple tokenization (placeholder)
    tokens = clean_text.split()
    
    # Language detection (placeholder)
    language = "vi" if any(char in "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ" for char in clean_text) else "en"
    
    return {
        "clean_text": clean_text,
        "language": language,
        "tokens": tokens
    }