"""End-to-end NLP preprocessing pipeline for crawled posts."""

from data_engine.core.dedup import text_hash
from data_engine.core.models import Platform, ProcessedRecord, RawPost
from data_engine.config.settings import get_settings
from data_engine.processing.language import detect_language
from data_engine.processing.normalizer import extract_emojis, light_clean, tokenize_preserve
from data_engine.processing.spam_filter import is_spam
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class ProcessingPipeline:
    """
    Transforms RawPost -> ProcessedRecord.
    Preserves raw text; cleaned text is for indexing only.
    """

    def __init__(self):
        self.settings = get_settings()

    def process(self, post: RawPost) -> ProcessedRecord | None:
        raw_text = post.text.strip()

        if len(raw_text) < self.settings.min_text_length:
            return None
        if len(raw_text) > self.settings.max_text_length:
            raw_text = raw_text[: self.settings.max_text_length]

        spam, spam_reason = is_spam(raw_text)
        if spam:
            logger.debug("spam_filtered", reason=spam_reason, platform=post.platform.value)
            return ProcessedRecord(
                text="",
                text_raw=raw_text,
                text_hash=text_hash(raw_text),
                language="mixed",
                source=post.platform,
                is_spam=True,
                metadata={"spam_reason": spam_reason},
            )

        cleaned = light_clean(raw_text)
        language = post.language_hint or detect_language(raw_text)
        if language not in ("vi", "en", "mixed"):
            language = detect_language(raw_text)

        emojis = extract_emojis(raw_text)
        tokens = tokenize_preserve(raw_text)

        return ProcessedRecord(
            text=cleaned if cleaned else raw_text,
            text_raw=raw_text,
            text_hash=text_hash(raw_text),
            language=language,
            source=post.platform,
            platform_post_id=post.post_id,
            url=post.url,
            timestamp=post.crawled_at,
            tokens=tokens,
            emojis=emojis,
            is_spam=False,
            metadata={
                **post.metadata,
                "author_id": post.author_id,
                "parent_id": post.parent_id,
            },
        )
