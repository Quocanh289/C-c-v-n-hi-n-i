from collections import Counter
from typing import Annotated, List, Optional
from urllib.parse import quote_plus
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.user_history import SaveTextRequest, UserSavedText, UserSavedTextResponse


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


async def get_db():
    async with async_session() as session:
        yield session


def normalize_issue(raw_issue: Optional[str]) -> str:
    if not raw_issue:
        return "Unknown"

    issue = raw_issue.strip()
    lowered = issue.lower()

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


def confidence_value(raw_confidence: Optional[float]) -> float:
    if raw_confidence is None:
        return 0.0
    return max(0.0, min(float(raw_confidence), 1.0))


def calculate_risk_level(records: List[UserSavedText]) -> str:
    if not records:
        return "Low"

    weighted_scores = []
    non_normal_count = 0
    has_suicidal_signal = False

    for record in records:
        issue = normalize_issue(record.predicted_issue)
        confidence = confidence_value(record.confidence)
        severity = ISSUE_SEVERITY.get(issue, 1 if issue != "Unknown" else 0)

        weighted_scores.append(severity * confidence)
        if issue not in {"Normal", "Unknown"}:
            non_normal_count += 1
        if issue == "Suicidal" and confidence >= 0.3:
            has_suicidal_signal = True

    average_weighted_score = sum(weighted_scores) / len(weighted_scores)
    highest_weighted_score = max(weighted_scores)
    non_normal_ratio = non_normal_count / len(records)

    if has_suicidal_signal:
        return "High"
    if highest_weighted_score >= 2.7 or average_weighted_score >= 1.7:
        return "High"
    if non_normal_ratio >= 0.6 and average_weighted_score >= 1.2:
        return "High"
    if highest_weighted_score >= 0.9 or average_weighted_score >= 0.6 or non_normal_ratio >= 0.3:
        return "Medium"
    return "Low"


def build_ai_summary(total: int, dominant_issue: str, average_confidence: float, risk_level: str) -> str:
    confidence_percent = round(average_confidence * 100)

    if total == 0:
        return (
            "Chưa có đủ dữ liệu đã lưu để tạo xu hướng cá nhân. "
            "Emotion Lens chỉ cung cấp thông tin tham khảo, không thay thế chẩn đoán hoặc tư vấn y khoa."
        )

    if risk_level == "High":
        recommendation = (
            f"Lịch sử gần đây có tín hiệu {dominant_issue} nổi bật với độ tin cậy trung bình khoảng "
            f"{confidence_percent}%, nên ưu tiên nghỉ ngơi, giảm tác nhân gây áp lực và cân nhắc trao đổi "
            "với chuyên gia sức khỏe tinh thần hoặc người đáng tin cậy."
        )
    elif risk_level == "Medium":
        recommendation = (
            f"Các đoạn đã lưu cho thấy {dominant_issue} xuất hiện thường xuyên nhất, với độ tin cậy trung bình "
            f"khoảng {confidence_percent}%. Bạn nên theo dõi xu hướng này, ghi nhận bối cảnh lặp lại và chủ động "
            "điều chỉnh nhịp sinh hoạt nếu cảm giác khó chịu kéo dài."
        )
    else:
        recommendation = (
            f"Tổng quan hiện ở mức thấp; nhãn nổi bật nhất là {dominant_issue} với độ tin cậy trung bình khoảng "
            f"{confidence_percent}%. Hãy tiếp tục quan sát cảm xúc theo thời gian để phát hiện thay đổi bất thường."
        )

    return (
        f"{recommendation} Đây không phải là chẩn đoán y khoa. Nếu bạn có ý nghĩ tự làm hại bản thân "
        "hoặc đang trong tình huống khẩn cấp, hãy liên hệ dịch vụ cấp cứu địa phương hoặc đường dây hỗ trợ khủng hoảng ngay."
    )


def aggregate_dashboard(records: List[UserSavedText]) -> AggregatedAnalysis:
    total_snippets = len(records)

    if total_snippets == 0:
        dominant_issue = "No data"
        average_confidence = 0.0
        risk_level = "Low"
    else:
        normalized_issues = [normalize_issue(record.predicted_issue) for record in records]
        dominant_issue = Counter(normalized_issues).most_common(1)[0][0]
        average_confidence = sum(confidence_value(record.confidence) for record in records) / total_snippets
        risk_level = calculate_risk_level(records)

    return AggregatedAnalysis(
        total_snippets=total_snippets,
        dominant_issue=dominant_issue,
        average_confidence=round(average_confidence, 4),
        risk_level=risk_level,
        ai_summary=build_ai_summary(total_snippets, dominant_issue, average_confidence, risk_level),
    )


@router.post(
    "/save-text",
    responses={
        400: {
            "description": "Mã UID không hợp lệ định dạng UUID",
            "content": {
                "application/json": {
                    "example": {"detail": "Mã UID không hợp lệ định dạng UUID"}
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
        raise HTTPException(status_code=400, detail="Mã UID không hợp lệ định dạng UUID")

    # TODO: Replace this with the real mental-health inference output.
    mock_prediction = "Stress"
    mock_confidence = 0.85

    new_record = UserSavedText(
        user_id=user_uuid,
        source_text=request.text,
        source_url=request.sourceUrl,
        predicted_issue=mock_prediction,
        confidence=mock_confidence,
    )

    db.add(new_record)
    await db.commit()
    await db.refresh(new_record)

    return {"status": "success", "message": "Đã lưu và phân tích văn bản thành công"}


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    responses={
        400: {
            "description": "Mã UID không hợp lệ định dạng UUID",
            "content": {
                "application/json": {
                    "example": {"detail": "Mã UID không hợp lệ định dạng UUID"}
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
        raise HTTPException(status_code=400, detail="Mã UID không hợp lệ định dạng UUID")

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
