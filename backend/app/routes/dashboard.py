from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Optional
from urllib.parse import quote_plus
import os
import sys
import uuid
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.user_history import SaveTextRequest, UserSavedText

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ai_nlp.analyzer import analyze_text


router = APIRouter(prefix="/api", tags=["Dashboard"])

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "emotionlens")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

DATABASE_URL = (
    f"postgresql+asyncpg://{quote_plus(POSTGRES_USER)}:{quote_plus(POSTGRES_PASSWORD)}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class EmotionNuance(BaseModel):
    key: str
    label: str
    score: float


class DistributionItem(BaseModel):
    key: str
    label: str
    ratio: float


class ConditionBadge(BaseModel):
    label: str
    severity: str


class AggregatedAnalysis(BaseModel):
    total_snippets: int
    dominant_issue: str
    average_confidence: float
    risk_level: str
    ai_summary: str
    emotion_distribution: list[DistributionItem] = Field(default_factory=list)
    issue_distribution: list[DistributionItem] = Field(default_factory=list)
    potential_conditions: list[ConditionBadge] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    disclaimer: str = ""


class DashboardSavedTextResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    source_text: str
    source_url: Optional[str] = None
    predicted_issue: Optional[str] = None
    confidence: Optional[float] = 0.0
    created_at: Any
    emotion_nuances: list[EmotionNuance] = Field(default_factory=list)


class DashboardResponse(BaseModel):
    saved_texts: list[DashboardSavedTextResponse]
    aggregated_analysis: AggregatedAnalysis


ISSUE_SEVERITY = {
    "Normal": 0,
    "Stress": 1,
    "Anxiety": 2,
    "Personality_disorder": 2,
    "Bipolar": 3,
    "Depression": 4,
    "Suicidal": 5,
}

SEVERITY_BASE_CONFIDENCE = {
    "low": 0.45,
    "medium": 0.62,
    "high": 0.82,
    "critical": 0.92,
}

EMOTION_TO_ISSUE = {
    "anxiety": "Anxiety",
    "fear": "Anxiety",
    "nervousness": "Anxiety",
    "sadness": "Depression",
    "grief": "Depression",
    "anger": "Stress",
    "annoyance": "Stress",
    "neutral": "Normal",
    "joy": "Normal",
    "admiration": "Normal",
    "love": "Normal",
    "surprise": "Stress",
}

EMOTION_LABELS = {
    "admiration": "Admiration",
    "amusement": "Amusement",
    "anger": "Anger",
    "annoyance": "Annoyance",
    "approval": "Approval",
    "caring": "Caring",
    "confusion": "Confusion",
    "curiosity": "Curiosity",
    "desire": "Desire",
    "disappointment": "Disappointment",
    "disapproval": "Disapproval",
    "disgust": "Disgust",
    "embarrassment": "Embarrassment",
    "excitement": "Excitement",
    "fear": "Fear",
    "gratitude": "Gratitude",
    "grief": "Grief",
    "joy": "Joy",
    "love": "Love",
    "nervousness": "Nervousness",
    "optimism": "Optimism",
    "pride": "Pride",
    "realization": "Realization",
    "relief": "Relief",
    "remorse": "Remorse",
    "sadness": "Sadness",
    "surprise": "Surprise",
    "neutral": "Neutral",
    "anxiety": "Anxiety",
}

KNOWN_ISSUE_ALIASES = {
    "tram cam": "Depression",
    "depression": "Depression",
    "lo au": "Anxiety",
    "anxiety": "Anxiety",
    "stress": "Stress",
    "cang thang": "Stress",
    "luong cuc": "Bipolar",
    "bipolar": "Bipolar",
    "tu tu": "Suicidal",
    "suicidal": "Suicidal",
    "personality disorder": "Personality_disorder",
    "personality_disorder": "Personality_disorder",
    "nhan cach": "Personality_disorder",
    "normal": "Normal",
    "healthy": "Normal",
}

NUANCE_ALIAS = {
    "anxious": "anxiety",
    "worried": "anxiety",
    "worry": "anxiety",
}

NEGATIVE_NUANCES = {
    "anger",
    "annoyance",
    "anxiety",
    "fear",
    "sadness",
    "grief",
    "disappointment",
    "disapproval",
    "disgust",
    "remorse",
    "nervousness",
    "embarrassment",
}

ANXIETY_NUANCES = {"anxiety", "nervousness", "fear", "confusion"}
DEPRESSIVE_NUANCES = {"sadness", "grief", "disappointment", "remorse", "embarrassment"}
ANGER_STRESS_NUANCES = {"anger", "annoyance", "disapproval", "disgust"}
SOCIAL_WITHDRAWAL_NUANCES = {"sadness", "grief", "fear", "embarrassment", "remorse"}

CONDITION_SEVERITY = {
    "Mild Stress": "low",
    "Anxiety Risk": "medium",
    "Depressive Symptoms": "medium",
    "Social Withdrawal Risk": "medium",
    "Burnout Risk": "high",
    "Acute Crisis Risk": "high",
    "No clear acute clinical risk pattern": "info",
}

DISCLAIMER_TEXT = (
    "Safety notice: this is not a medical diagnosis. If you feel at risk of harming yourself or someone else, "
    "contact local emergency services immediately (911 in the United States) or your local crisis hotline."
)


async def get_db():
    async with async_session() as session:
        yield session


async def ensure_emotion_nuances_column(db: AsyncSession) -> bool:
    check_query = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'user_saved_texts' AND column_name = 'emotion_nuances'
        LIMIT 1
        """
    )

    result = await db.execute(check_query)
    if result.scalar() == 1:
        return True

    try:
        await db.execute(
            text(
                """
                ALTER TABLE user_saved_texts
                ADD COLUMN IF NOT EXISTS emotion_nuances JSONB NOT NULL DEFAULT '[]'::jsonb
                """
            )
        )
        await db.commit()
        return True
    except Exception:
        await db.rollback()
        return False


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def confidence_value(raw_confidence: Optional[float]) -> float:
    if raw_confidence is None:
        return 0.0
    return max(0.0, min(float(raw_confidence), 1.0))


def normalize_issue(raw_issue: Optional[str]) -> str:
    if not raw_issue:
        return "Unknown"

    issue = raw_issue.strip()
    lowered = issue.lower()

    if lowered in KNOWN_ISSUE_ALIASES:
        return KNOWN_ISSUE_ALIASES[lowered]

    if "suicid" in lowered or "tu tu" in lowered:
        return "Suicidal"
    if "depress" in lowered or "tram cam" in lowered:
        return "Depression"
    if "anxiety" in lowered or "anxious" in lowered or "lo au" in lowered:
        return "Anxiety"
    if "bipolar" in lowered or "luong cuc" in lowered:
        return "Bipolar"
    if "stress" in lowered or "cang thang" in lowered:
        return "Stress"
    if "personality" in lowered or "nhan cach" in lowered:
        return "Personality_disorder"
    if "normal" in lowered or "healthy" in lowered or "binh thuong" in lowered:
        return "Normal"

    return issue


def normalize_nuance(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None

    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    key = NUANCE_ALIAS.get(key, key)

    if key in EMOTION_LABELS:
        return key

    if key in {"positive", "calm", "peaceful"}:
        return "neutral"

    return None


def infer_from_conditions(conditions: Any) -> tuple[Optional[str], Optional[float]]:
    if not isinstance(conditions, list):
        return None, None

    best_issue = None
    best_confidence = -1.0

    for item in conditions:
        if not isinstance(item, dict):
            continue

        raw_name = item.get("name") or item.get("label") or item.get("condition")
        if not raw_name:
            continue

        issue = normalize_issue(str(raw_name))
        score = confidence_value(to_float(item.get("confidence", item.get("score", item.get("probability", 0.0)))))

        if score > best_confidence:
            best_confidence = score
            best_issue = issue

    if best_issue is None:
        return None, None

    return best_issue, best_confidence


def infer_from_all_scores(all_scores: Any) -> tuple[Optional[str], Optional[float]]:
    if not isinstance(all_scores, dict) or not all_scores:
        return None, None

    best_issue = None
    best_confidence = -1.0

    for raw_label, raw_score in all_scores.items():
        issue = normalize_issue(str(raw_label))
        score = confidence_value(to_float(raw_score))
        if score > best_confidence:
            best_confidence = score
            best_issue = issue

    if best_issue is None:
        return None, None

    return best_issue, best_confidence


def infer_from_emotions_and_severity(analysis_result: dict[str, Any]) -> tuple[str, float]:
    emotions = analysis_result.get("emotions")
    primary_emotion = None
    if isinstance(emotions, list) and emotions:
        primary_emotion = str(emotions[0]).lower()
    elif isinstance(emotions, str):
        primary_emotion = emotions.lower()

    issue = EMOTION_TO_ISSUE.get(primary_emotion or "", "Unknown")

    severity = str(analysis_result.get("severity", "medium")).lower()
    confidence = SEVERITY_BASE_CONFIDENCE.get(severity, 0.55)

    sentiment = str(analysis_result.get("sentiment", "neutral")).lower()
    if sentiment == "negative":
        confidence += 0.06
        if issue == "Unknown":
            issue = "Stress"

    risk_signals = analysis_result.get("risk_signals")
    if isinstance(risk_signals, list):
        confidence += min(len(risk_signals), 3) * 0.03
        if len(risk_signals) >= 2 and issue == "Unknown":
            issue = "Depression"

    issue = normalize_issue(issue)
    if issue == "Unknown":
        issue = "Normal"

    return issue, confidence_value(confidence)


def extract_prediction_from_analysis(analysis_result: Any) -> tuple[str, float]:
    if not isinstance(analysis_result, dict):
        return "Normal", 0.35

    explicit_pairs = [
        ("predicted_issue", "confidence"),
        ("primary_condition", "primary_confidence"),
        ("condition", "condition_confidence"),
        ("diagnosis", "confidence"),
    ]

    for issue_key, confidence_key in explicit_pairs:
        raw_issue = analysis_result.get(issue_key)
        if raw_issue:
            issue = normalize_issue(str(raw_issue))
            confidence = confidence_value(to_float(analysis_result.get(confidence_key, 0.0), default=0.0))
            if confidence > 0:
                return issue, confidence

    condition_issue, condition_confidence = infer_from_conditions(
        analysis_result.get("possible_related_conditions")
    )
    if condition_issue and condition_confidence is not None:
        return condition_issue, confidence_value(condition_confidence)

    score_issue, score_confidence = infer_from_all_scores(analysis_result.get("all_scores"))
    if score_issue and score_confidence is not None:
        return score_issue, confidence_value(score_confidence)

    return infer_from_emotions_and_severity(analysis_result)


def extract_emotion_nuances(analysis_result: Any) -> list[EmotionNuance]:
    if not isinstance(analysis_result, dict):
        return [EmotionNuance(key="neutral", label="Neutral", score=0.35)]

    raw_scores: dict[str, float] = {}
    score_keys = ["scores_28", "all_scores", "scores_9", "scores", "emotion_scores"]

    for key in score_keys:
        candidate = analysis_result.get(key)
        if not isinstance(candidate, dict):
            continue
        for raw_label, raw_score in candidate.items():
            nuance_key = normalize_nuance(str(raw_label))
            if nuance_key is None:
                continue
            score = confidence_value(to_float(raw_score))
            if score <= 0:
                continue
            raw_scores[nuance_key] = max(raw_scores.get(nuance_key, 0.0), score)

    if not raw_scores:
        emotions = analysis_result.get("emotions")
        if isinstance(emotions, list):
            for raw_label in emotions:
                nuance_key = normalize_nuance(str(raw_label))
                if nuance_key:
                    raw_scores[nuance_key] = max(raw_scores.get(nuance_key, 0.0), 0.55)
        elif isinstance(emotions, str):
            nuance_key = normalize_nuance(emotions)
            if nuance_key:
                raw_scores[nuance_key] = 0.55

    if not raw_scores:
        primary = analysis_result.get("primary_emotion")
        nuance_key = normalize_nuance(primary if isinstance(primary, str) else None)
        if nuance_key:
            raw_scores[nuance_key] = confidence_value(to_float(analysis_result.get("confidence", 0.5), default=0.5))

    if not raw_scores:
        return [EmotionNuance(key="neutral", label="Neutral", score=0.35)]

    sorted_items = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    top_items = sorted_items[:4]

    return [
        EmotionNuance(
            key=key,
            label=EMOTION_LABELS.get(key, key.replace("_", " ").title()),
            score=round(score, 4),
        )
        for key, score in top_items
    ]


def parse_emotion_nuances_from_db(raw_value: Any) -> list[EmotionNuance]:
    if raw_value is None:
        return []

    parsed = raw_value
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return []

    if not isinstance(parsed, list):
        return []

    nuances: list[EmotionNuance] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        key = normalize_nuance(str(item.get("key") or item.get("emotion") or item.get("label") or ""))
        if not key:
            continue
        score = confidence_value(to_float(item.get("score", item.get("confidence", 0.0))))
        nuances.append(
            EmotionNuance(
                key=key,
                label=EMOTION_LABELS.get(key, key.replace("_", " ").title()),
                score=round(score, 4),
            )
        )

    return nuances


async def save_emotion_nuances_to_db(
    db: AsyncSession,
    record_id: uuid.UUID,
    nuances: list[EmotionNuance],
) -> None:
    is_ready = await ensure_emotion_nuances_column(db)
    if not is_ready:
        return

    payload = json.dumps([nuance.model_dump() for nuance in nuances])
    await db.execute(
        text(
            """
            UPDATE user_saved_texts
            SET emotion_nuances = CAST(:nuances AS jsonb)
            WHERE id = :record_id
            """
        ),
        {
            "record_id": record_id,
            "nuances": payload,
        },
    )
    await db.commit()


async def load_emotion_nuances_map(db: AsyncSession, user_id: uuid.UUID) -> dict[str, list[EmotionNuance]]:
    if not await ensure_emotion_nuances_column(db):
        return {}

    result = await db.execute(
        text(
            """
            SELECT id, emotion_nuances
            FROM user_saved_texts
            WHERE user_id = :user_id
            """
        ),
        {"user_id": user_id},
    )

    nuances_map: dict[str, list[EmotionNuance]] = {}
    for row in result.mappings().all():
        row_id = str(row.get("id"))
        nuances_map[row_id] = parse_emotion_nuances_from_db(row.get("emotion_nuances"))

    return nuances_map


def calculate_risk_level(records: list[DashboardSavedTextResponse]) -> str:
    if not records:
        return "Low"

    weighted_scores = []
    non_normal_count = 0
    high_severity_count = 0
    has_suicidal_signal = False

    for record in records:
        issue = normalize_issue(record.predicted_issue)
        confidence = confidence_value(record.confidence)
        severity = ISSUE_SEVERITY.get(issue, 1 if issue != "Unknown" else 0)

        weighted_scores.append(severity * confidence)
        if issue not in {"Normal", "Unknown"}:
            non_normal_count += 1
        if severity >= 3:
            high_severity_count += 1
        if issue == "Suicidal" and confidence >= 0.3:
            has_suicidal_signal = True

    average_weighted_score = sum(weighted_scores) / len(weighted_scores)
    highest_weighted_score = max(weighted_scores)
    non_normal_ratio = non_normal_count / len(records)
    high_severity_ratio = high_severity_count / len(records)

    if has_suicidal_signal:
        return "High"
    if highest_weighted_score >= 2.6 or average_weighted_score >= 1.7:
        return "High"
    if high_severity_ratio >= 0.35:
        return "High"
    if non_normal_ratio >= 0.6 and average_weighted_score >= 1.15:
        return "High"
    if highest_weighted_score >= 0.9 or average_weighted_score >= 0.6 or non_normal_ratio >= 0.3:
        return "Medium"
    return "Low"


def summarize_emotion_distribution(records: list[DashboardSavedTextResponse]) -> tuple[Counter[str], dict[str, float]]:
    weighted_counts: Counter[str] = Counter()

    for record in records:
        for nuance in record.emotion_nuances:
            weighted_counts[nuance.key] += max(0.05, nuance.score)

    if not weighted_counts:
        weighted_counts["neutral"] = 1.0

    total_weight = sum(weighted_counts.values()) or 1.0
    ratio_map = {key: weighted_counts[key] / total_weight for key in weighted_counts}

    return weighted_counts, ratio_map


def build_issue_distribution(issue_counts: Counter[str], total: int) -> list[DistributionItem]:
    if total <= 0 or not issue_counts:
        return [DistributionItem(key="normal", label="Normal", ratio=1.0)]

    return [
        DistributionItem(
            key=issue.lower(),
            label=issue.replace("_", " ").title(),
            ratio=round(count / total, 4),
        )
        for issue, count in issue_counts.most_common(6)
    ]


def build_emotion_distribution(
    records: list[DashboardSavedTextResponse],
) -> tuple[list[DistributionItem], dict[str, float], Counter[str]]:
    emotion_counts, nuance_ratios = summarize_emotion_distribution(records)
    top_items = emotion_counts.most_common(6)

    distribution = [
        DistributionItem(
            key=key,
            label=EMOTION_LABELS.get(key, key.replace("_", " ").title()),
            ratio=round(nuance_ratios.get(key, 0.0), 4),
        )
        for key, _ in top_items
    ]

    return distribution, nuance_ratios, emotion_counts


def build_condition_badges(conditions: list[str]) -> list[ConditionBadge]:
    return [
        ConditionBadge(label=condition, severity=CONDITION_SEVERITY.get(condition, "info"))
        for condition in conditions
    ]


def build_recommendations(risk_level: str, negative_load: float) -> list[str]:
    if risk_level == "High":
        recommendations = [
            "Reduce overload today and prioritize rest blocks.",
            "Avoid isolation and share your current state with a trusted person.",
            "Schedule a mental health consultation this week for a structured assessment.",
        ]
    elif risk_level == "Medium":
        recommendations = [
            "Keep a daily check-in and track triggers alongside recovery habits.",
            "Protect sleep quality and add short regulation routines (breathing, movement, journaling).",
            "Seek professional support if symptoms intensify or persist beyond two weeks.",
        ]
    else:
        recommendations = [
            "Maintain protective routines and continue logging snippets for trend tracking.",
            "Watch for sudden increases in fear, anxiety, or sadness.",
            "Balance workload with recovery breaks to sustain stability.",
        ]

    if negative_load >= 0.55:
        recommendations.append(
            "The negative-emotion load is elevated, so frequent recovery breaks are strongly recommended."
        )

    return recommendations


def infer_potential_conditions(
    total: int,
    issue_counts: Counter[str],
    nuance_ratios: dict[str, float],
    non_normal_ratio: float,
    high_severity_ratio: float,
    risk_level: str,
) -> list[str]:
    potential_conditions: list[str] = []

    def ratio_for(keys: set[str]) -> float:
        return sum(nuance_ratios.get(key, 0.0) for key in keys)

    anxiety_ratio = ratio_for(ANXIETY_NUANCES)
    depressive_ratio = ratio_for(DEPRESSIVE_NUANCES)
    stress_ratio = ratio_for(ANGER_STRESS_NUANCES)
    withdrawal_ratio = ratio_for(SOCIAL_WITHDRAWAL_NUANCES)

    if stress_ratio >= 0.22 or issue_counts.get("Stress", 0) / total >= 0.25:
        potential_conditions.append("Mild Stress")

    if anxiety_ratio >= 0.24 or issue_counts.get("Anxiety", 0) / total >= 0.2:
        potential_conditions.append("Anxiety Risk")

    if depressive_ratio >= 0.24 or issue_counts.get("Depression", 0) / total >= 0.16:
        potential_conditions.append("Depressive Symptoms")

    if withdrawal_ratio >= 0.28 and non_normal_ratio >= 0.45:
        potential_conditions.append("Social Withdrawal Risk")

    if high_severity_ratio >= 0.28 or (risk_level == "High" and non_normal_ratio >= 0.5):
        potential_conditions.append("Burnout Risk")

    if issue_counts.get("Suicidal", 0) > 0:
        potential_conditions.append("Acute Crisis Risk")

    if not potential_conditions:
        potential_conditions.append("No clear acute clinical risk pattern")

    return potential_conditions


def build_ai_summary(
    total: int,
    dominant_issue: str,
    dominant_ratio: float,
    average_confidence: float,
    risk_level: str,
    issue_distribution: list[DistributionItem],
    emotion_distribution: list[DistributionItem],
    non_normal_ratio: float,
    high_severity_ratio: float,
    potential_conditions: list[str],
    recommendations: list[str],
) -> str:
    if total == 0:
        return (
            "No saved snippets yet, so a personalized trend cannot be computed. "
            "This dashboard is informational only and cannot replace professional diagnosis or treatment."
        )

    confidence_percent = round(average_confidence * 100)
    issue_distribution_text = ", ".join(
        f"{item.label}: {round(item.ratio * 100)}%" for item in issue_distribution
    )
    emotion_distribution_text = ", ".join(
        f"{item.label}: {round(item.ratio * 100)}%" for item in emotion_distribution
    )
    potential_conditions_text = ", ".join(potential_conditions)
    recommendation_text = " ".join(recommendations)

    return (
        f"Across {total} saved snippets, the dominant mental-health label is {dominant_issue} "
        f"({round(dominant_ratio * 100)}%), with average model confidence at {confidence_percent}%. "
        f"Issue distribution: {issue_distribution_text}. Emotion nuance trend: {emotion_distribution_text}. "
        f"Non-normal ratio: {round(non_normal_ratio * 100)}%; higher-severity ratio: {round(high_severity_ratio * 100)}%. "
        f"Potential conditions you might be experiencing: {potential_conditions_text}. "
        f"Recommendation: {recommendation_text} "
        f"{DISCLAIMER_TEXT}"
    )


def aggregate_dashboard(records: list[DashboardSavedTextResponse]) -> AggregatedAnalysis:
    total_snippets = len(records)
    if total_snippets == 0:
        empty_issue_distribution = [DistributionItem(key="normal", label="Normal", ratio=1.0)]
        empty_emotion_distribution = [DistributionItem(key="neutral", label="Neutral", ratio=1.0)]
        empty_conditions = [ConditionBadge(label="No data yet", severity="info")]
        empty_recommendations = [
            "Save a few snippets to unlock personalized recommendations.",
            "Keep logging short reflections to capture trends over time.",
        ]

        return AggregatedAnalysis(
            total_snippets=0,
            dominant_issue="No data",
            average_confidence=0.0,
            risk_level="Low",
            ai_summary=build_ai_summary(
                total=0,
                dominant_issue="No data",
                dominant_ratio=0.0,
                average_confidence=0.0,
                risk_level="Low",
                issue_distribution=empty_issue_distribution,
                emotion_distribution=empty_emotion_distribution,
                non_normal_ratio=0.0,
                high_severity_ratio=0.0,
                potential_conditions=[condition.label for condition in empty_conditions],
                recommendations=empty_recommendations,
            ),
            emotion_distribution=empty_emotion_distribution,
            issue_distribution=empty_issue_distribution,
            potential_conditions=empty_conditions,
            recommendations=empty_recommendations,
            disclaimer=DISCLAIMER_TEXT,
        )

    normalized_issues = [normalize_issue(record.predicted_issue) for record in records]
    issue_counts = Counter(normalized_issues)
    dominant_issue = issue_counts.most_common(1)[0][0]
    dominant_ratio = issue_counts.get(dominant_issue, 0) / total_snippets

    confidence_scores = [confidence_value(record.confidence) for record in records]
    average_confidence = sum(confidence_scores) / total_snippets

    non_normal_ratio = sum(1 for issue in normalized_issues if issue not in {"Normal", "Unknown"}) / total_snippets
    high_severity_ratio = (
        sum(1 for issue in normalized_issues if ISSUE_SEVERITY.get(issue, 0) >= 3) / total_snippets
    )

    risk_level = calculate_risk_level(records)
    issue_distribution = build_issue_distribution(issue_counts, total_snippets)
    emotion_distribution, nuance_ratios, _ = build_emotion_distribution(records)

    potential_conditions = infer_potential_conditions(
        total=total_snippets,
        issue_counts=issue_counts,
        nuance_ratios=nuance_ratios,
        non_normal_ratio=non_normal_ratio,
        high_severity_ratio=high_severity_ratio,
        risk_level=risk_level,
    )
    condition_badges = build_condition_badges(potential_conditions)
    negative_load = sum(nuance_ratios.get(key, 0.0) for key in NEGATIVE_NUANCES)
    recommendations = build_recommendations(risk_level, negative_load)

    ai_summary = build_ai_summary(
        total=total_snippets,
        dominant_issue=dominant_issue,
        dominant_ratio=dominant_ratio,
        average_confidence=average_confidence,
        risk_level=risk_level,
        issue_distribution=issue_distribution,
        emotion_distribution=emotion_distribution,
        non_normal_ratio=non_normal_ratio,
        high_severity_ratio=high_severity_ratio,
        potential_conditions=potential_conditions,
        recommendations=recommendations,
    )

    return AggregatedAnalysis(
        total_snippets=total_snippets,
        dominant_issue=dominant_issue,
        average_confidence=round(average_confidence, 4),
        risk_level=risk_level,
        ai_summary=ai_summary,
        emotion_distribution=emotion_distribution,
        issue_distribution=issue_distribution,
        potential_conditions=condition_badges,
        recommendations=recommendations,
        disclaimer=DISCLAIMER_TEXT,
    )


@router.post(
    "/save-text",
    responses={
        400: {
            "description": "UID is not a valid UUID format",
            "content": {
                "application/json": {
                    "example": {"detail": "UID is not a valid UUID format"}
                }
            },
        }
    },
)
async def save_highlighted_text(
    request: SaveTextRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        user_uuid = uuid.UUID(request.uid)
    except ValueError:
        raise HTTPException(status_code=400, detail="UID is not a valid UUID format")

    input_text = getattr(request, "source_text", None) or request.text

    try:
        analysis_result = analyze_text(input_text)
        predicted_issue, confidence = extract_prediction_from_analysis(analysis_result)
        emotion_nuances = extract_emotion_nuances(analysis_result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(exc)}")

    new_record = UserSavedText(
        user_id=user_uuid,
        source_text=request.text,
        source_url=request.sourceUrl,
        predicted_issue=predicted_issue,
        confidence=confidence,
    )

    db.add(new_record)
    await db.commit()
    await db.refresh(new_record)

    try:
        await save_emotion_nuances_to_db(db, new_record.id, emotion_nuances)
    except Exception:
        await db.rollback()

    return {
        "status": "success",
        "message": "Text snippet saved and analyzed successfully",
        "predicted_issue": predicted_issue,
        "confidence": confidence,
        "emotion_nuances": [nuance.model_dump() for nuance in emotion_nuances],
    }


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    responses={
        400: {
            "description": "UID is not a valid UUID format",
            "content": {
                "application/json": {
                    "example": {"detail": "UID is not a valid UUID format"}
                }
            },
        }
    },
)
async def get_user_dashboard(
    uid: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        user_uuid = uuid.UUID(uid)
    except ValueError:
        raise HTTPException(status_code=400, detail="UID is not a valid UUID format")

    query = (
        select(UserSavedText)
        .where(UserSavedText.user_id == user_uuid)
        .order_by(UserSavedText.created_at.desc())
    )
    result = await db.execute(query)
    records = list(result.scalars().all())

    nuance_map = await load_emotion_nuances_map(db, user_uuid)

    dashboard_rows: list[DashboardSavedTextResponse] = []

    for record in records:
        record_nuances = nuance_map.get(str(record.id), [])

        if not record_nuances:
            try:
                derived_nuances = extract_emotion_nuances(analyze_text(record.source_text))
                await save_emotion_nuances_to_db(db, record.id, derived_nuances)
                record_nuances = derived_nuances
            except Exception:
                record_nuances = [EmotionNuance(key="neutral", label="Neutral", score=0.35)]

        dashboard_rows.append(
            DashboardSavedTextResponse(
                id=record.id,
                user_id=record.user_id,
                source_text=record.source_text,
                source_url=record.source_url,
                predicted_issue=record.predicted_issue,
                confidence=confidence_value(record.confidence),
                created_at=record.created_at,
                emotion_nuances=record_nuances,
            )
        )

    return DashboardResponse(
        saved_texts=dashboard_rows,
        aggregated_analysis=aggregate_dashboard(dashboard_rows),
    )
