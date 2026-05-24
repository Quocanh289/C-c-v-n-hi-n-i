# ====================================================
# Analysis Routes - Core Emotion Detection API
# Uses the trained 28-label GoEmotions model
# 28 fine-grained for English, 9 coarse for Vietnamese
# ====================================================

import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends, Request

from app.models.inference import GoEmotionsInference, get_inference, GOEMOTIONS_28, COARSE_EMOTIONS
from app.models.mental_health_inference import get_mental_health_inference
# --- THÊM ĐOẠN NÀY VÀO ĐỂ ĐƯA ĐƯỜNG DẪN HỆ THỐNG LÊN THƯ MỤC GỐC ---
import sys
import os
# Đi từ backend/app/routes/ lên 3 cấp (../../../) để chạm tới thư mục gốc dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# --- BÂY GIỜ CÁC LỆNH IMPORT AI_NLP SẼ HOẠT ĐỘNG HOÀN HẢO ---
from ai_nlp.emotion_model import COARSE_EMOTIONS
from ai_nlp.analyzer import analyze_text  # Hàm xử lý chính trong file analyzer.py
from ai_nlp.training.emotion_pipeline.config import GOEMOTIONS_28

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


# ====================================================
# Pydantic Models
# ====================================================

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
    label_type: str = "fine"  # "fine" (28) or "coarse" (9)
    language: str = "en"
    scores_28: Dict[str, float] = Field(default_factory=dict)  # 28 fine-grained scores (for English)
    scores_9: Dict[str, float] = Field(default_factory=dict)   # 9 coarse scores (for Vietnamese)
    toxicity_score: float = 0.0
    toxicity_binary: bool = False
    sarcasm_score: float = 0.0
    sarcasm_binary: bool = False
    source: str = "goemotions_28"
    model: str = "xlm-roberta-base+lora+goemotions28"
    num_labels: int = 28
    processing_time_ms: float = 0.0
    diagnosis_code: str = "normal"
    diagnosis_vi: str = "Không có dấu hiệu rõ ràng"
    risk_level: str = "none"
    needs_attention: bool = False
    mental_health_screening: Dict[str, Any] = Field(default_factory=dict)


class BatchAnalyzeResponse(BaseModel):
    """Response model for batch analysis."""
    results: List[AnalyzeResponse]
    total_processing_time_ms: float = 0.0


# ====================================================
# Routes
# ====================================================

@router.post("", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest):
    """
    Analyze a single text for emotions using the 28-label GoEmotions model.
    
    For English text (auto-detected): returns 28 fine-grained emotion scores
    For Vietnamese text (auto-detected): returns 9 coarse emotion scores
    
    Vietnamese uses aggregated scores from the 28-label model.
    Dedicated Vietnamese training data coming soon.
    """
    start = time.time()
    
    try:
        infer = get_inference()
        result = infer.classify(
            text=request.text,
            output_mode=request.output_mode,  # None = auto
        )
        screening = get_mental_health_inference().classify(request.text)
        # Gọi trực tiếp hàm phân tích văn bản thực tế từ module ai_nlp
        result = analyze_text(request.text)
        
        processing_time = (time.time() - start) * 1000
        
        # result trả về là một dictionary chứa thông tin phân tích
        primary_emotion_val = "neutral"
        confidence_val = 0.0
        label_type_val = "fine"
        language_val = "en"
        scores_28_val = {}
        scores_9_val = {}

        if isinstance(result, dict):
            primary_emotion_val = result.get("primary_emotion", "neutral")
            confidence_val = result.get("confidence", 0.0)
            label_type_val = result.get("label_type", "fine")
            language_val = result.get("language", "en")
            scores_28_val = result.get("scores_28", {})
            scores_9_val = result.get("scores_9", {})

        return AnalyzeResponse(
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            primary_emotion=result.get("primary_emotion", "neutral"),
            confidence=result.get("confidence", 0.0),
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
            processing_time_ms=processing_time,
            diagnosis_code=screening.get("diagnosis_code", "normal"),
            diagnosis_vi=screening.get("diagnosis_vi", "Không có dấu hiệu rõ ràng"),
            risk_level=screening.get("risk_level", "none"),
            needs_attention=screening.get("needs_attention", False),
            mental_health_screening=screening,
            primary_emotion=primary_emotion_val,
            confidence=confidence_val,
            label_type=label_type_val,
            language=language_val,
            scores_28=scores_28_val,
            scores_9=scores_9_val,
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    """
    Analyze multiple texts in batch.
    
    Each text is analyzed independently with language auto-detection.
    English → 28 fine-grained labels, Vietnamese → 9 coarse labels.
    """
    start = time.time()
    
    try:
        results = []
        for text in request.texts:
            result = infer.classify(text=text, output_mode=None)
            result["mental_health_screening"] = get_mental_health_inference().classify(text)
            # Gọi trực tiếp hàm phân tích của ai_nlp cho từng đoạn văn bản
            result = analyze_text(text)
            results.append(result)
        
        response_results = [
            AnalyzeResponse(
                text=t[:100] + "..." if len(t) > 100 else t,
                primary_emotion=r.get("primary_emotion", "neutral") if isinstance(r, dict) else "neutral",
                confidence=r.get("confidence", 0.0) if isinstance(r, dict) else 0.0,
                label_type=r.get("label_type", "fine") if isinstance(r, dict) else "fine",
                language=r.get("language", "en") if isinstance(r, dict) else "en",
                scores_28=r.get("scores_28", {}) if isinstance(r, dict) else {},
                scores_9=r.get("scores_9", {}) if isinstance(r, dict) else {},
                toxicity_score=r.get("toxicity_score", 0.0) if isinstance(r, dict) else 0.0,
                toxicity_binary=r.get("toxicity_binary", False) if isinstance(r, dict) else False,
                sarcasm_score=r.get("sarcasm_score", 0.0) if isinstance(r, dict) else 0.0,
                sarcasm_binary=r.get("sarcasm_binary", False) if isinstance(r, dict) else False,
                source=r.get("source", "goemotions_28") if isinstance(r, dict) else "goemotions_28",
                model=r.get("model", "xlm-roberta-base+lora+goemotions28") if isinstance(r, dict) else "xlm-roberta-base+lora+goemotions28",
                num_labels=r.get("num_labels", 28) if isinstance(r, dict) else 28,
                processing_time_ms=0.0,
                diagnosis_code=r["mental_health_screening"].get("diagnosis_code", "normal"),
                diagnosis_vi=r["mental_health_screening"].get("diagnosis_vi", "Không có dấu hiệu rõ ràng"),
                risk_level=r["mental_health_screening"].get("risk_level", "none"),
                needs_attention=r["mental_health_screening"].get("needs_attention", False),
                mental_health_screening=r["mental_health_screening"],
            )
            for t, r in zip(request.texts, results)
        ]
        
        return BatchAnalyzeResponse(
            results=response_results,
            total_processing_time_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/labels")
async def get_labels():
    """
    Get available emotion labels.
    Returns 28 fine-grained labels for English and 9 coarse for Vietnamese.
    """
    return {
        "english": {
            "num_labels": 28,
            "labels": GOEMOTIONS_28,
            "description": "28 fine-grained GoEmotions labels (trained model)",
        },
        "vietnamese": {
            "num_labels": 9,
            "labels": COARSE_EMOTIONS,
            "description": "9 coarse emotions (aggregated from 28-label model - no Vietnamese training data yet)",
            "note": "Dedicated Vietnamese emotion dataset and training coming soon",
        },
    }
