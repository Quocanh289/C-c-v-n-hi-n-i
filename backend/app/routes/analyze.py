# ====================================================
# Analysis Routes - Core Emotion Detection API
# Uses GoEmotions-shaped emotion inference.
# Vietnamese text is translated to English before detection.
# ====================================================

import logging
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.inference import COARSE_EMOTIONS, GOEMOTIONS_28, get_inference
from app.models.mental_health_inference import get_mental_health_inference
from app.routes.translate import contains_vietnamese, translate_simple

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


class AnalyzeRequest(BaseModel):
    """Request model for emotion analysis."""

    text: str = Field(..., min_length=1, max_length=2000, description="Text to analyze")
    return_all_probs: bool = Field(True, description="Return probabilities for all emotions")
    output_mode: Optional[str] = Field(None, description="'fine' (28), 'coarse' (9), or 'auto' (28 EN, 9 VI)")


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch analysis."""

    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to analyze")
    return_all_probs: bool = Field(True, description="Return probabilities for all emotions")


class AnalyzeResponse(BaseModel):
    """Response model for emotion analysis."""

    text: str
    primary_emotion: str
    confidence: float = 0.0
    top_emotions: List[Dict[str, Union[str, float]]] = Field(default_factory=list)
    label_type: str = "fine"
    language: str = "en"
    scores_28: Dict[str, float] = Field(default_factory=dict)
    scores_9: Dict[str, float] = Field(default_factory=dict)
    toxicity_score: float = 0.0
    toxicity_binary: bool = False
    sarcasm_score: float = 0.0
    sarcasm_binary: bool = False
    source: str = "goemotions_28"
    model: str = "xlm-roberta-base+lora+goemotions28"
    num_labels: int = 28
    processing_time_ms: float = 0.0
    diagnosis_code: str = "normal"
    diagnosis_vi: str = "Khong co dau hieu ro rang"
    risk_level: str = "none"
    needs_attention: bool = False
    mental_health_screening: Dict[str, Any] = Field(default_factory=dict)
    translation: Dict[str, str] = Field(default_factory=dict)


class BatchAnalyzeResponse(BaseModel):
    """Response model for batch analysis."""

    results: List[AnalyzeResponse]
    total_processing_time_ms: float = 0.0


def _prepare_text(text: str) -> tuple[str, Dict[str, str]]:
    if not contains_vietnamese(text):
        return text, {}
    translated = translate_simple(text)
    return translated, {
        "source_language": "vi",
        "target_language": "en",
        "translated_text": translated,
    }


def _to_response(original_text: str, result: Dict[str, Any], screening: Dict[str, Any], processing_time_ms: float) -> AnalyzeResponse:
    return AnalyzeResponse(
        text=original_text[:100] + "..." if len(original_text) > 100 else original_text,
        primary_emotion=result.get("primary_emotion", "neutral"),
        confidence=result.get("confidence", 0.0),
        top_emotions=result.get("top_emotions", []),
        label_type=result.get("label_type", "fine"),
        language=result.get("language", "en"),
        scores_28=result.get("scores_28", {}),
        scores_9=result.get("scores_9", {}),
        toxicity_score=result.get("toxicity_score", 0.0),
        toxicity_binary=result.get("toxicity_binary", False),
        sarcasm_score=result.get("sarcasm_score", 0.0),
        sarcasm_binary=result.get("sarcasm_binary", False),
        source=result.get("source", "goemotions_28"),
        model=result.get("model", "xlm-roberta-base+lora+goemotions28"),
        num_labels=result.get("num_labels", 28),
        processing_time_ms=processing_time_ms,
        diagnosis_code=screening.get("diagnosis_code", "normal"),
        diagnosis_vi=screening.get("diagnosis_vi", "Khong co dau hieu ro rang"),
        risk_level=screening.get("risk_level", "none"),
        needs_attention=screening.get("needs_attention", False),
        mental_health_screening=screening,
        translation=result.get("translation", {}),
    )


@router.post("", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest):
    """Analyze one text. Vietnamese text is translated to English first."""

    start = time.time()

    try:
        text_for_analysis, translation = _prepare_text(request.text)
        if translation:
            logger.info("Translated VI->EN for emotion analysis: %s -> %s", request.text[:60], text_for_analysis[:60])

        result = get_inference().classify(text=text_for_analysis, output_mode=request.output_mode)
        if translation:
            result["language"] = "vi"
            result["label_type"] = "fine"
            result["num_labels"] = 28
            result["translation"] = translation

        screening = get_mental_health_inference().classify(text_for_analysis)
        return _to_response(request.text, result, screening, (time.time() - start) * 1000)
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    """Analyze multiple texts. Vietnamese texts are translated to English first."""

    start = time.time()

    try:
        infer = get_inference()
        response_results: List[AnalyzeResponse] = []
        for text in request.texts:
            text_for_analysis, translation = _prepare_text(text)
            result = infer.classify(text=text_for_analysis, output_mode=None)
            if translation:
                result["language"] = "vi"
                result["label_type"] = "fine"
                result["num_labels"] = 28
                result["translation"] = translation
            screening = get_mental_health_inference().classify(text_for_analysis)
            response_results.append(_to_response(text, result, screening, 0.0))

        return BatchAnalyzeResponse(
            results=response_results,
            total_processing_time_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        logger.error("Batch analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/labels")
async def get_labels():
    """Get available emotion labels."""

    return {
        "english": {
            "num_labels": 28,
            "labels": GOEMOTIONS_28,
            "description": "28 fine-grained GoEmotions labels",
        },
        "vietnamese": {
            "num_labels": 28,
            "labels": GOEMOTIONS_28,
            "description": "Vietnamese text is translated to English, then analyzed with GoEmotions labels",
            "legacy_coarse_labels": COARSE_EMOTIONS,
        },
    }
