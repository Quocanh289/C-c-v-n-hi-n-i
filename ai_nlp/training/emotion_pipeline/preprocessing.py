"""
Preprocessing Module
=====================
Text preprocessing for Reddit-style internet language.
Handles emojis, slang, repeated characters, URLs, Unicode normalization,
and other real-world text artifacts.

Designed for robustness on informal English social-media text.
"""

import re
import emoji
import logging
import unicodedata
from typing import Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# Text Normalizers
# ============================================================

class RedditTextNormalizer:
    """
    Normalizes Reddit-style internet text for emotion detection.
    
    Handles:
    - Unicode normalization (NFKC)
    - Emoji extraction and representation
    - Repeated character normalization ("sooo good" → "so good")
    - URL and mention removal
    - HTML entity decoding
    - Contraction expansion (optional)
    - Slang normalization (optional)
    - Excessive whitespace
    """
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        handle_emojis: bool = True,
        handle_repeated_chars: bool = True,
        handle_urls: bool = True,
        handle_mentions: bool = True,
        expand_contractions: bool = False,
        slang_to_formal: bool = False,
        max_repeated_chars: int = 3,
    ):
        self.normalize_unicode = normalize_unicode
        self.handle_emojis = handle_emojis
        self.handle_repeated_chars = handle_repeated_chars
        self.handle_urls = handle_urls
        self.handle_mentions = handle_mentions
        self.expand_contractions = expand_contractions
        self.slang_to_formal = slang_to_formal
        self.max_repeated_chars = max_repeated_chars
        
        # Pre-compiled patterns
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+|bit\.ly/\S+|shorturl\.\S+', re.IGNORECASE
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.repeated_char_pattern = re.compile(r'(.)\1{%d,}' % (max_repeated_chars - 1))
        self.html_entity_pattern = re.compile(r'&[a-zA-Z]+;')
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s\']')
        
        # Emoji-related
        self._emoji_pattern = None
        if handle_emojis:
            self._emoji_pattern = re.compile(
                "[" 
                "\U0001F600-\U0001F64F"  # Emoticons
                "\U0001F300-\U0001F5FF"  # Symbols & pictographs
                "\U0001F680-\U0001F6FF"  # Transport & map
                "\U0001F1E0-\U0001F1FF"  # Flags
                "\U00002702-\U000027B0"  # Dingbats
                "\U000024C2-\U0001F251"  # Enclosed
                "\U0001F900-\U0001F9FF"  # Supplemental
                "\U0001FA00-\U0001FA6F"  # Chess symbols
                "\U0001FA70-\U0001FAFF"  # Symbols extended-A
                "\U00002600-\U000026FF"  # Miscellaneous
                "\U0000FE00-\U0000FE0F"  # Variation selectors
                "\U0000200D"             # Zero width joiner
                "]+", flags=re.UNICODE
            )
    
    def normalize(self, text: str) -> str:
        """
        Apply all configured normalizations to input text.
        Returns cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # HTML entities
        text = self._normalize_html_entities(text)
        
        # Extract emoji information (before removal)
        emoji_info = ""
        if self.handle_emojis and self._emoji_pattern:
            emojis_found = self._emoji_pattern.findall(text)
            if emojis_found:
                emoji_info = " ".join(emojis_found)
                text = self._emoji_pattern.sub(" ", text)
        
        # URLs
        if self.handle_urls:
            text = self.url_pattern.sub("[URL]", text)
        
        # Mentions
        if self.handle_mentions:
            text = self.mention_pattern.sub("[USER]", text)
        
        # Repeated characters
        if self.handle_repeated_chars:
            text = self._normalize_repeated_chars(text)
        
        # Contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        # Slang
        if self.slang_to_formal:
            text = self._normalize_slang(text)
        
        # Clean up extra whitespace
        text = self.whitespace_pattern.sub(" ", text).strip()
        
        # Append emoji info if present (as features, not tokens)
        if emoji_info:
            text = f"{text} {emoji_info}"
        
        return text
    
    def _normalize_html_entities(self, text: str) -> str:
        """Convert common HTML entities to characters."""
        entity_map = {
            '&': '&', '<': '<', '>': '>',
            '"': '"', '&#39;': "'", '&#x27;': "'",
            '&#x2F;': '/', '&#x60;': '`', '&#x3D;': '=',
            '&nbsp;': ' ', '&ndash;': '-', '&mdash;': '--',
            '&hellip;': '...', '&rsquo;': "'", '&lsquo;': "'",
            '&ldquo;': '"', '&rdquo;': '"',
        }
        for entity, char in entity_map.items():
            text = text.replace(entity, char)
        # Fallback for unknown entities
        text = self.html_entity_pattern.sub("", text)
        return text
    
    def _normalize_repeated_chars(self, text: str) -> str:
        """
        Normalize repeated characters like "sooooo" → "sooo".
        Preserves at most max_repeated_chars of the same character.
        
        Special handling for common internet patterns:
        - "soooo good" → "soo good" (emphasis preserved with double)
        - "noooooo" → "nooo" (keeps some repetition for emphasis)
        - "!!!!!!" → "!!!" (keeps some)
        """
        def replace_repeated(match):
            char = match.group(1)
            count = len(match.group(0))
            if char in "!?.,;:":
                # Keep at most 3 punctuation repeats
                return char * min(count, 3)
            elif char.lower() in "aeiou":
                # Keep at most 3 vowel repeats (preserves emotional emphasis)
                return char * min(count, 3)
            else:
                # Keep at most 2 consonant repeats
                return char * min(count, 2)
        
        return self.repeated_char_pattern.sub(replace_repeated, text)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        contractions = {
            r"\bain't\b": "is not",
            r"\baren't\b": "are not",
            r"\bcan't\b": "cannot",
            r"\bcouldn't\b": "could not",
            r"\bdidn't\b": "did not",
            r"\bdoesn't\b": "does not",
            r"\bdon't\b": "do not",
            r"\bhadn't\b": "had not",
            r"\bhasn't\b": "has not",
            r"\bhaven't\b": "have not",
            r"\bhe'll\b": "he will",
            r"\bhe's\b": "he is",
            r"\bi'd\b": "i would",
            r"\bi'll\b": "i will",
            r"\bi'm\b": "i am",
            r"\bi've\b": "i have",
            r"\bisn't\b": "is not",
            r"\bit's\b": "it is",
            r"\bLet's\b": "let us",
            r"\bmightn't\b": "might not",
            r"\bmustn't\b": "must not",
            r"\bshan't\b": "shall not",
            r"\bshe'll\b": "she will",
            r"\bshe's\b": "she is",
            r"\bshouldn't\b": "should not",
            r"\bthat's\b": "that is",
            r"\bthere's\b": "there is",
            r"\bthey'd\b": "they would",
            r"\bthey'll\b": "they will",
            r"\bthey're\b": "they are",
            r"\bthey've\b": "they have",
            r"\bwasn't\b": "was not",
            r"\bwe'd\b": "we would",
            r"\bwe'll\b": "we will",
            r"\bwe're\b": "we are",
            r"\bwe've\b": "we have",
            r"\bweren't\b": "were not",
            r"\bwhat's\b": "what is",
            r"\bwon't\b": "will not",
            r"\bwouldn't\b": "would not",
            r"\byou'd\b": "you would",
            r"\byou'll\b": "you will",
            r"\byou're\b": "you are",
            r"\byou've\b": "you have",
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def _normalize_slang(self, text: str) -> str:
        """Normalize common internet slang to more formal equivalents."""
        slang_map = {
            r"\baf\b": "as fuck",
            r"\basf\b": "as fuck",
            r"\bbruh\b": "bro",
            r"\bcap\b": "lie",
            r"\bdead\b(?!\s*$|\.|!)": "dying laughing",
            r"\bfr\b": "for real",
            r"\bgoat\b": "greatest of all time",
            r"\blit\b": "exciting",
            r"\blol\b": "laughing",
            r"\blmao\b": "laughing",
            r"\blmfao\b": "laughing",
            r"\bnah\b": "no",
            r"\bno cap\b": "no lie",
            r"\bong\b": "on god",
            r"\bsus\b": "suspicious",
            r"\btbh\b": "to be honest",
            r"\btf\b": "the fuck",
            r"\bw\b(?=\s|$)": "win",
            r"\bya\b": "you",
            r"\byo\b": "you",
            r"\byeah\b": "yes",
            r"\byep\b": "yes",
            r"\bya'll\b": "you all",
            r"\bgonna\b": "going to",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bdunno\b": "do not know",
            r"\bimma\b": "i am going to",
            r"\bcuz\b": "because",
            r"\bcos\b": "because",
            r"\bcoz\b": "because",
            r"\btho\b": "though",
            r"\bpls\b": "please",
            r"\bplz\b": "please",
            r"\brly\b": "really",
        }
        for pattern, replacement in slang_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def extract_emoji_sentiment(self, text: str) -> Tuple[List[str], float]:
        """
        Extract emojis from text and compute a simple sentiment score.
        
        Returns:
            Tuple of (list of emoji strings, sentiment score [-1, 1])
        """
        emoji_sentiment_map = {
            # Positive
            "😀": 0.8, "😃": 0.8, "😄": 0.9, "😁": 0.8, "😆": 0.9,
            "😅": 0.6, "😂": 0.7, "🤣": 0.9, "😊": 1.0, "😇": 0.9,
            "🙂": 0.5, "😌": 0.7, "😍": 1.0, "🥰": 1.0, "😘": 0.9,
            "😗": 0.7, "😋": 0.8, "😛": 0.6, "😜": 0.7, "🤪": 0.5,
            "😝": 0.6, "🤑": 0.5, "🤗": 0.9, "🤩": 1.0, "😎": 0.7,
            "🥳": 1.0, "🔥": 0.8, "💯": 0.9, "✨": 0.6, "❤️": 1.0,
            "💕": 0.9, "💖": 1.0, "💗": 1.0, "💙": 0.8, "💚": 0.8,
            "💛": 0.8, "💜": 0.8, "🤍": 0.5, "👍": 0.7, "🙌": 0.9,
            "👏": 0.7, "🎉": 1.0, "🎊": 1.0, "⭐": 0.6, "🌟": 0.7,
            "💪": 0.6,
            # Negative
            "😞": -0.7, "😔": -0.7, "😟": -0.5, "😕": -0.3, "🙁": -0.5,
            "☹️": -0.6, "😣": -0.6, "😖": -0.7, "😫": -0.6, "😩": -0.6,
            "😤": -0.6, "😠": -0.8, "😡": -0.9, "🤬": -1.0, "😢": -0.8,
            "😭": -0.9, "😨": -0.7, "😰": -0.7, "😱": -0.6, "😳": -0.2,
            "🤯": -0.3, "😒": -0.4, "🙄": -0.4, "😏": -0.2, "👎": -0.6,
            "💔": -0.8, "💢": -0.7, "💀": -0.3, "☠️": -0.4,
            # Ambiguous
            "🤔": 0.0, "🤷": 0.0, "🤨": -0.1, "😐": 0.0, "😑": 0.0,
        }
        
        found_emojis = []
        sentiment = 0.0
        count = 0
        
        for char in text:
            if char in emoji_sentiment_map:
                found_emojis.append(char)
                sentiment += emoji_sentiment_map[char]
                count += 1
        
        if count > 0:
            sentiment /= count
        
        return found_emojis, sentiment
    
    def get_preprocessing_stats(self, text: str) -> Dict:
        """
        Get preprocessing statistics for analysis.
        Useful for understanding data distribution.
        """
        stats = {
            "original_length": len(text),
            "has_url": bool(self.url_pattern.search(text)),
            "has_mention": bool(self.mention_pattern.search(text)),
            "has_emoji": bool(self._emoji_pattern.search(text)) if self._emoji_pattern else False,
            "has_repeated": bool(self.repeated_char_pattern.search(text)),
            "word_count_original": len(text.split()),
        }
        normalized = self.normalize(text)
        stats["normalized_length"] = len(normalized)
        stats["word_count_normalized"] = len(normalized.split())
        stats["normalized"] = normalized
        return stats


# ============================================================
# Multi-Label Preprocessing Pipeline
# ============================================================

class MultiLabelPreprocessor:
    """
    Complete preprocessing pipeline for multi-label GoEmotions data.
    Combines text normalization with label processing.
    """
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        handle_emojis: bool = True,
        handle_repeated_chars: bool = True,
        handle_urls: bool = True,
    ):
        self.text_normalizer = RedditTextNormalizer(
            normalize_unicode=normalize_unicode,
            handle_emojis=handle_emojis,
            handle_repeated_chars=handle_repeated_chars,
            handle_urls=handle_urls,
        )
    
    def process_text(self, text: str) -> str:
        """Normalize a single text for training/inference."""
        return self.text_normalizer.normalize(text)
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Normalize a batch of texts."""
        return [self.process_text(t) for t in texts]
    
    def process_labels(self, multi_hot: List[int], as_coarse: bool = False) -> List[int]:
        """
        Process multi-hot labels.
        
        Args:
            multi_hot: 28-element binary vector (GoEmotions labels)
            as_coarse: If True, aggregate to 9 coarse labels
        
        Returns:
            Processed label vector
        """
        from .config import EMOTION_27_TO_9_MAP, COARSE_TO_IDX, GOEMOTIONS_28
        
        if not as_coarse:
            # Return as-is for fine-grained
            return multi_hot
        
        # Aggregate to coarse
        coarse = [0] * 9
        for idx, val in enumerate(multi_hot):
            if val == 0:
                continue
            fine_label = GOEMOTIONS_28[idx]
            if fine_label == "neutral":
                coarse[COARSE_TO_IDX["neutral"]] = 1
            elif fine_label in EMOTION_27_TO_9_MAP:
                coarse_label = EMOTION_27_TO_9_MAP[fine_label]
                coarse[COARSE_TO_IDX[coarse_label]] = 1
        
        return coarse


# ============================================================
# Robustness Evaluation Tools
# ============================================================

class RobustnessEvaluator:
    """
    Evaluates model robustness on internet language challenges.
    Tests: slang, emojis, repeated chars, sarcasm, informal English, meme language.
    """
    
    def __init__(self, model_predict_fn: Callable):
        """
        Args:
            model_predict_fn: Function that takes text and returns label predictions
        """
        self.predict = model_predict_fn
    
    def test_slang(self, samples: List[str]) -> Dict:
        """Test robustness to internet slang."""
        results = []
        for text in samples:
            pred = self.predict(text)
            results.append({"text": text, "prediction": pred})
        return {"task": "slang", "samples": results}
    
    def test_emojis(self, samples: List[str]) -> Dict:
        """Test robustness to emoji-rich text."""
        results = []
        for text in samples:
            pred = self.predict(text)
            results.append({"text": text, "prediction": pred})
        return {"task": "emojis", "samples": results}
    
    def test_repeated_chars(self, samples: List[str]) -> Dict:
        """Test robustness to repeated character emphasis."""
        results = []
        for text in samples:
            pred = self.predict(text)
            results.append({"text": text, "prediction": pred})
        return {"task": "repeated_chars", "samples": results}
    
    def test_sarcasm(self, samples: List[str]) -> Dict:
        """Test robustness to sarcastic phrasing."""
        results = []
        for text in samples:
            pred = self.predict(text)
            results.append({"text": text, "prediction": pred})
        return {"task": "sarcasm", "samples": results}
    
    def evaluate_all(self) -> Dict:
        """Run all robustness tests."""
        internet_samples = {
            "slang": [
                "bro cooked 💀",
                "nah this is insane 😭",
                "W take honestly",
                "no cap fr fr",
                "this lit af 🔥",
                "ong this is fire",
                "that's based as hell",
                "im dead 💀💀💀",
                "this goes hard",
                "W rizz no cap",
            ],
            "emojis": [
                "That was amazing 😍🔥💯",
                "I'm so over this 💀🙄",
                "Best day ever 🎉🥳✨",
                "🤔🤷 not sure about this",
                "💔😭 why would you do this",
                "😡🤬 absolutely furious",
                "❤️💕 this is so sweet",
                "🥺👉👈 please?",
            ],
            "repeated_chars": [
                "Nooooooo why would you do that",
                "That's sooooo goooood omgggg",
                "I hate this so muchhhhh",
                "hiiiiiiiiiii how are youuuuu",
                "Whaaaaaat no waaaay",
                "I'm sooooo tiiiiired",
                "yesssssss finallyyyyy",
                "😭😭😭😭😭😭😭",
            ],
            "sarcasm": [
                "great... just great 🙂",
                "oh wow, another meeting, fantastic",
                "i love how nothing works around here",
                "yeah, because that's worked out SO well for us before",
                "oh sure, blame the person who actually fixed it",
                "my favorite thing is being ignored, really",
                "wow, thanks for stating the obvious, genius",
                "oh i'm SO sorry for having feelings",
            ],
        }
        
        results = {}
        for task, samples in internet_samples.items():
            results[task] = self._run_robustness_test(task, samples)
        
        return results
    
    def _run_robustness_test(self, task: str, samples: List[str]) -> Dict:
        results = []
        for text in samples:
            pred = self.predict(text)
            results.append({"text": text, "prediction": pred})
        return {"task": task, "n_samples": len(samples), "results": results}