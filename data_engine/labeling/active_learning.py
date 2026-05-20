"""Active learning sampler — prioritizes uncertain examples for labeling."""

from typing import Optional

from data_engine.config.settings import get_settings


class ActiveLearningSampler:
    """
    Confidence-based sampling for semi-automatic labeling.
    Prioritizes samples in the uncertainty band (e.g. 0.35–0.65).
    """

    def __init__(
        self,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
    ):
        settings = get_settings()
        self.low = low_threshold or settings.active_learning_confidence_low
        self.high = high_threshold or settings.active_learning_confidence_high

    def priority_score(
        self,
        confidence: float,
        is_disagreement: bool = False,
    ) -> float:
        """
        Higher score = label sooner.
        Peak uncertainty at midpoint of band.
        """
        if is_disagreement:
            return 1.0

        if confidence < self.low or confidence > self.high:
            return 0.1  # high-confidence predictions deprioritized

        # Uncertainty band — highest priority near 0.5
        midpoint = (self.low + self.high) / 2
        distance = abs(confidence - midpoint)
        band_width = (self.high - self.low) / 2
        return 1.0 - (distance / band_width)

    def select_batch(
        self,
        samples: list[dict],
        batch_size: int = 50,
    ) -> list[dict]:
        """Select top-priority samples for human/AI labeling."""
        scored = [
            (self.priority_score(s.get("confidence", 0.5), s.get("is_disagreement", False)), s)
            for s in samples
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:batch_size]]
