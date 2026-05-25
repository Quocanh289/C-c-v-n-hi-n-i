from collections import Counter
from pathlib import Path
from typing import Annotated, Any, List, Optional
from urllib.parse import quote_plus
import os
import sys
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.user_history import SaveTextRequest, UserSavedText, UserSavedTextResponse

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


class AggregatedAnalysis(BaseModel):
    total_snippets: int
    dominant_issue: str
    average_confidence: float
    risk_level: str
    ai_summary: str


class DashboardResponse(BaseModel):
    saved_texts: List[UserSavedTextResponse]
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
    "sadness": "Depression",
    "anger": "Stress",
    "neutral": "Normal",
    "joy": "Normal",
    "admiration": "Normal",
    "love": "Normal",
    "surprise": "Stress",
}

KNOWN_ISSUE_ALIASES = {
    "tram cam": "Depression",
    "trầm cảm": "Depression",
    "depression": "Depression",
    "lo au": "Anxiety",
    "lo âu": "Anxiety",
    "anxiety": "Anxiety",
    "stress": "Stress",
    "cang thang": "Stress",
    "căng thẳng": "Stress",
    "luong cuc": "Bipolar",
    "lưỡng cực": "Bipolar",
    "bipolar": "Bipolar",
    "tu tu": "Suicidal",
    "tự tử": "Suicidal",
    "suicidal": "Suicidal",
    "personality disorder": "Personality_disorder",
    "personality_disorder": "Personality_disorder",
    "nhan cach": "Personality_disorder",
    "nhân cách": "Personality_disorder",
    "normal": "Normal",
    "healthy": "Normal",
}


async def get_db():
    async with async_session() as session:
        yield session


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

    if "suicid" in lowered or "tự tử" in lowered or "tu tu" in lowered:
        return "Suicidal"
    if "depress" in lowered or "trầm cảm" in lowered or "tram cam" in lowered:
        return "Depression"
    if "anxiety" in lowered or "anxious" in lowered or "lo âu" in lowered or "lo au" in lowered:
        return "Anxiety"
    if "bipolar" in lowered or "lưỡng cực" in lowered or "luong cuc" in lowered:
        return "Bipolar"
    if "stress" in lowered or "căng thẳng" in lowered or "cang thang" in lowered:
        return "Stress"
    if "personality" in lowered or "nhân cách" in lowered or "nhan cach" in lowered:
        return "Personality_disorder"
    if "normal" in lowered or "healthy" in lowered or "bình thường" in lowered or "binh thuong" in lowered:
        return "Normal"

    return issue


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


def calculate_risk_level(records: List[UserSavedText]) -> str:
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


def build_ai_summary(
    total: int,
    dominant_issue: str,
    average_confidence: float,
    risk_level: str,
    issue_counts: Counter[str],
    non_normal_ratio: float,
    high_severity_ratio: float,
) -> str:
    if total == 0:
        return (
            "Chua co du du lieu da luu de tao xu huong ca nhan. "
            "Emotion Lens chi cung cap thong tin tham khao, khong thay the chan doan hoac tu van y khoa."
        )

    confidence_percent = round(average_confidence * 100)
    dominant_ratio = (issue_counts.get(dominant_issue, 0) / total) if total > 0 else 0.0

    top_issues = issue_counts.most_common(3)
    distribution_text = ", ".join(
        f"{issue}: {round((count / total) * 100)}%"
        for issue, count in top_issues
    )

    if average_confidence >= 0.75:
        confidence_note = "Do tin cay du doan dang o muc cao."
    elif average_confidence >= 0.55:
        confidence_note = "Do tin cay du doan dang o muc trung binh."
    else:
        confidence_note = "Do tin cay du doan dang o muc thap, nen theo doi them de tang do chac chan."

    if risk_level == "High":
        action_note = (
            "Can uu tien giam ap luc, tranh co lap va tim ho tro som tu nguoi than hoac chuyen gia "
            "suc khoe tinh than."
        )
    elif risk_level == "Medium":
        action_note = (
            "Nen duy tri theo doi hang ngay, ghi nhan tinh huong kich hoat cam xuc va dieu chinh nhip sinh hoat."
        )
    else:
        action_note = "Hien tai xu huong tuong doi on dinh, tiep tuc duy tri thoi quen tu cham soc ban than."

    summary_main = (
        f"Trong {total} doan van ban da luu, van de xuat hien nhieu nhat la {dominant_issue} "
        f"({round(dominant_ratio * 100)}%), do tin cay trung binh {confidence_percent}%. "
        f"Phan bo hien tai: {distribution_text}. {confidence_note} "
        f"Ty le van de ngoai 'Normal' la {round(non_normal_ratio * 100)}%, "
        f"nhom muc do nang la {round(high_severity_ratio * 100)}%."
    )

    return (
        f"{summary_main} {action_note} "
        "Day khong phai chan doan y khoa. Neu ban co y nghi tu lam hai ban than "
        "hoac dang trong tinh huong khan cap, hay lien he dich vu cap cuu dia phuong ngay lap tuc."
    )


def aggregate_dashboard(records: List[UserSavedText]) -> AggregatedAnalysis:
    total_snippets = len(records)
    if total_snippets == 0:
        return AggregatedAnalysis(
            total_snippets=0,
            dominant_issue="No data",
            average_confidence=0.0,
            risk_level="Low",
            ai_summary=build_ai_summary(
                total=0,
                dominant_issue="No data",
                average_confidence=0.0,
                risk_level="Low",
                issue_counts=Counter(),
                non_normal_ratio=0.0,
                high_severity_ratio=0.0,
            ),
        )

    normalized_issues = [normalize_issue(record.predicted_issue) for record in records]
    issue_counts = Counter(normalized_issues)
    dominant_issue = issue_counts.most_common(1)[0][0]

    confidence_scores = [confidence_value(record.confidence) for record in records]
    average_confidence = sum(confidence_scores) / total_snippets

    non_normal_ratio = sum(1 for issue in normalized_issues if issue not in {"Normal", "Unknown"}) / total_snippets
    high_severity_ratio = (
        sum(1 for issue in normalized_issues if ISSUE_SEVERITY.get(issue, 0) >= 3) / total_snippets
    )

    risk_level = calculate_risk_level(records)
    ai_summary = build_ai_summary(
        total=total_snippets,
        dominant_issue=dominant_issue,
        average_confidence=average_confidence,
        risk_level=risk_level,
        issue_counts=issue_counts,
        non_normal_ratio=non_normal_ratio,
        high_severity_ratio=high_severity_ratio,
    )

    return AggregatedAnalysis(
        total_snippets=total_snippets,
        dominant_issue=dominant_issue,
        average_confidence=round(average_confidence, 4),
        risk_level=risk_level,
        ai_summary=ai_summary,
    )


@router.post(
    "/save-text",
    responses={
        400: {
            "description": "Ma UID khong hop le dinh dang UUID",
            "content": {
                "application/json": {
                    "example": {"detail": "Ma UID khong hop le dinh dang UUID"}
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
        raise HTTPException(status_code=400, detail="Ma UID khong hop le dinh dang UUID")

    input_text = getattr(request, "source_text", None) or request.text

    try:
        analysis_result = analyze_text(input_text)
        predicted_issue, confidence = extract_prediction_from_analysis(analysis_result)
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

    return {
        "status": "success",
        "message": "Da luu va phan tich van ban thanh cong",
        "predicted_issue": predicted_issue,
        "confidence": confidence,
    }


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    responses={
        400: {
            "description": "Ma UID khong hop le dinh dang UUID",
            "content": {
                "application/json": {
                    "example": {"detail": "Ma UID khong hop le dinh dang UUID"}
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
        raise HTTPException(status_code=400, detail="Ma UID khong hop le dinh dang UUID")

    query = (
        select(UserSavedText)
        .where(UserSavedText.user_id == user_uuid)
        .order_by(UserSavedText.created_at.desc())
    )
    result = await db.execute(query)
    records = list(result.scalars().all())

    return DashboardResponse(
        saved_texts=[UserSavedTextResponse.model_validate(record) for record in records],
        aggregated_analysis=aggregate_dashboard(records),
    )
