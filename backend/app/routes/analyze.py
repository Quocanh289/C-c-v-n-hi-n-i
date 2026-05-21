# ====================================================
# Analysis Routes - Core Emotion Detection API
# Uses the trained 28-label GoEmotions model
# 28 fine-grained for English, 9 coarse for Vietnamese
# ====================================================

import logging
import time
from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, Depends, Request

from app.models.inference import GoEmotionsInference, get_inference, GOEMOTIONS_28, COARSE_EMOTIONS

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
    scores_28: dict = {}  # 28 fine-grained scores (for English)
    scores_9: dict = {}   # 9 coarse scores (for Vietnamese)
    toxicity_score: float = 0.0
    toxicity_binary: bool = False
    sarcasm_score: float = 0.0
    sarcasm_binary: bool = False
    source: str = "goemotions_28"
    model: str = "xlm-roberta-base+lora+goemotions28"
    num_labels: int = 28
    processing_time_ms: float = 0.0


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
        
        processing_time = (time.time() - start) * 1000
        
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
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
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
        infer = get_inference()
        results = []
        
        for text in request.texts:
            result = infer.classify(text=text, output_mode=None)
            results.append(result)
        
        response_results = [
            AnalyzeResponse(
                text=t[:100] + "..." if len(t) > 100 else t,
                primary_emotion=r.get("primary_emotion", "neutral"),
                confidence=r.get("confidence", 0.0),
                label_type=r.get("label_type", "fine"),
                language=r.get("language", "en"),
                scores_28=r.get("scores_28", {}),
                scores_9=r.get("scores_9", {}),
                toxicity_score=r.get("toxicity_score", 0.0),
                toxicity_binary=r.get("toxicity_binary", False),
                sarcasm_score=r.get("sarcasm_score", 0.0),
                sarcasm_binary=r.get("sarcasm_binary", False),
                source=r.get("source", "goemotions_28"),
                model=r.get("model", "xlm-roberta-base+lora+goemotions28"),
                num_labels=r.get("num_labels", 28),
                processing_time_ms=0.0,
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