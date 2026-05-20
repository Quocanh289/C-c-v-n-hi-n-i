"""Label Studio integration for multi-annotator workflows."""

from typing import Any, Optional

import httpx

from data_engine.config.settings import get_settings
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class LabelStudioClient:
    """Push tasks to Label Studio and pull completed annotations."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.label_studio_url or "").rstrip("/")
        self.api_key = api_key or settings.label_studio_api_key
        self.headers = {"Authorization": f"Token {self.api_key}"} if self.api_key else {}

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key)

    async def import_tasks(
        self,
        project_id: int,
        tasks: list[dict[str, Any]],
    ) -> dict:
        if not self.enabled:
            logger.warning("label_studio_not_configured")
            return {"imported": 0}

        url = f"{self.base_url}/api/projects/{project_id}/import"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                headers=self.headers,
                json=tasks,
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()

    def build_emotion_task(self, record: dict) -> dict:
        """Standard Label Studio task format for emotion labeling."""
        return {
            "data": {
                "text": record["text"],
                "language": record.get("language", "mixed"),
                "source": record.get("source", "unknown"),
                "record_id": record.get("id"),
            },
        }

    def compute_consensus(
        self,
        annotations: list[dict],
        field: str = "emotion",
    ) -> dict:
        """
        Majority vote consensus across annotators.
        Returns label + agreement score.
        """
        votes: dict[str, int] = {}
        for ann in annotations:
            label = ann.get("result", [{}])[0].get("value", {}).get(field)
            if label:
                votes[label] = votes.get(label, 0) + 1

        if not votes:
            return {"label": None, "agreement": 0.0}

        winner = max(votes, key=votes.get)
        agreement = votes[winner] / len(annotations)
        return {"label": winner, "agreement": agreement, "votes": votes}
