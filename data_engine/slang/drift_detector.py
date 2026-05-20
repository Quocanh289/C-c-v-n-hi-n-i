"""Slang drift detection — tracks token frequency changes over time."""

from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

import re

from data_engine.config.settings import get_settings
from data_engine.core.models import SlangCandidate
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)

# Tokens to ignore (common words)
STOPWORDS_VI = {
    "là", "của", "và", "có", "không", "một", "được", "trong", "cho",
    "với", "này", "đó", "như", "để", "rất", "cũng", "nhưng", "thì",
}
STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "and", "in", "that", "it", "for", "on", "with",
}

SLANG_TOKEN_PATTERN = re.compile(
    r"[a-zA-ZàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ0-9]+",
    re.UNICODE,
)


class SlangDriftDetector:
    """
    Compares token frequencies between time windows.
    Flags high-growth unknown tokens as emerging slang candidates.
    """

    def __init__(self, known_terms: Optional[set[str]] = None):
        self.settings = get_settings()
        self.known_terms = known_terms or set()
        self.stopwords = STOPWORDS_VI | STOPWORDS_EN

    def extract_tokens(self, text: str) -> list[str]:
        tokens = SLANG_TOKEN_PATTERN.findall(text.lower())
        return [
            t for t in tokens
            if len(t) >= 2
            and t not in self.stopwords
            and not t.isdigit()
        ]

    def build_frequency_map(
        self,
        texts: list[str],
    ) -> Counter:
        counter: Counter = Counter()
        for text in texts:
            counter.update(self.extract_tokens(text))
        return counter

    def detect_emerging(
        self,
        current_texts: list[str],
        previous_texts: list[str],
        top_n: int = 50,
    ) -> list[SlangCandidate]:
        """
        Compare current vs previous window frequencies.
        Returns candidates sorted by growth rate.
        """
        current_freq = self.build_frequency_map(current_texts)
        previous_freq = self.build_frequency_map(previous_texts)

        candidates: list[SlangCandidate] = []

        for term, curr_count in current_freq.most_common(top_n * 3):
            if term in self.known_terms:
                continue
            if curr_count < self.settings.slang_min_frequency:
                continue

            prev_count = previous_freq.get(term, 0)
            if prev_count == 0:
                growth = float(curr_count)  # new term
            else:
                growth = curr_count / prev_count

            if growth < self.settings.slang_growth_threshold:
                continue

            contexts = [
                t[:200] for t in current_texts
                if term in t.lower()
            ][:5]

            candidates.append(
                SlangCandidate(
                    term=term,
                    frequency_current=curr_count,
                    frequency_previous=prev_count,
                    growth_rate=growth,
                    contexts=contexts,
                    status="emerging" if growth >= self.settings.slang_growth_threshold * 2 else "candidate",
                )
            )

        candidates.sort(key=lambda c: c.growth_rate, reverse=True)
        logger.info(
            "slang_drift_scan",
            candidates_found=len(candidates),
            top_term=candidates[0].term if candidates else None,
        )
        return candidates[:top_n]

    def is_likely_slang(self, term: str, context: str) -> bool:
        """Heuristic: short novel tokens in informal context."""
        if term in self.known_terms or term in self.stopwords:
            return False
        informal_markers = ["💀", "😭", "fr", "lol", "bro", "vcl", "trời", "ảo"]
        return any(m in context.lower() for m in informal_markers)
