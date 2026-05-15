# ====================================================
# Analysis Routes - Core Emotion Detection API
# ====================================================

import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, Depends, Request

from app.models.emotion_model import InferenceResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


# ====================================================
# Pydantic Models
# ====================================================

class AnalyzeRequest(BaseModel):
    """Request model for emotion analysis."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to analyze")
    return_all_probs: bool = Field(False, description="Return probabilities for all emotions")


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch analysis."""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to analyze")
    return_all_probs: bool = Field(False, description="Return probabilities for all emotions")


class AnalyzeResponse(BaseModel):
    """Response model for emotion analysis."""
    text: str
    primary_emotion: str
    emotions: dict = {}
    toxicity_score: float = 0.0
    toxicity_binary: bool = False
    sarcasm_score: float = 0.0
    sarcasm_binary: bool = False
    intent: str = "unknown"
    confidence: float = 0.0
    language: str = "en"
    source: str = "backend"
    processing_time_ms: float = 0.0


class BatchAnalyzeResponse(BaseModel):
    """Response model for batch analysis."""
    results: List[AnalyzeResponse]
    total_processing_time_ms: float = 0.0
    cache_stats: dict = {}


# ====================================================
# Dependency: Get model manager from app state
# ====================================================

async def get_model_manager(request: Request):
    """Get the EmotionModelManager instance from FastAPI app state."""
    from app.main import model_manager
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return model_manager


# ====================================================
# Routes
# ====================================================

@router.post("", response_model=AnalyzeResponse)
async def analyze_text(
    request: AnalyzeRequest,
    manager=Depends(get_model_manager),
):
    """
    Analyze a single text for emotions.
    
    Uses the multi-task XLM-RoBERTa model to detect:
    - Primary emotion (joy, anger, sadness, anxiety, fear, surprise, neutral, toxic, sarcastic)
    - Toxicity level
    - Sarcasm level
    - Intent
    - Language
    
    Returns confidence scores and timings.
    """
    try:
        result: InferenceResult = manager.analyze(
            text=request.text,
            return_all_probs=request.return_all_probs,
        )
        
        return AnalyzeResponse(
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            primary_emotion=result.primary_emotion,
            emotions=result.emotions if request.return_all_probs else {},
            toxicity_score=result.toxicity_score,
            toxicity_binary=result.toxicity_binary,
            sarcasm_score=result.sarcasm_score,
            sarcasm_binary=result.sarcasm_binary,
            intent=result.intent,
            confidence=result.confidence,
            language=result.language,
            source="backend",
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(
    request: BatchAnalyzeRequest,
    manager=Depends(get_model_manager),
):
    """
    Analyze multiple texts in batch for efficiency.
    
    Processes up to 100 texts at once, leveraging GPU batching
    for maximum throughput. Returns individual results plus
    cache statistics.
    """
    import time
    start = time.time()
    
    try:
        results = manager.analyze_batch(
            texts=request.texts,
            batch_size=min(len(request.texts), 32),
        )
        
        response_results = [
            AnalyzeResponse(
                text=text[:100] + "..." if len(text) > 100 else text,
                primary_emotion=r.primary_emotion,
                emotions=r.emotions if request.return_all_probs else {},
                toxicity_score=r.toxicity_score,
                toxicity_binary=r.toxicity_binary,
                sarcasm_score=r.sarcasm_score,
                sarcasm_binary=r.sarcasm_binary,
                intent=r.intent,
                confidence=r.confidence,
                language=r.language,
                source="backend",
                processing_time_ms=r.processing_time_ms,
            )
            for text, r in zip(request.texts, results)
        ]
        
        return BatchAnalyzeResponse(
            results=response_results,
            total_processing_time_ms=(time.time() - start) * 1000,
            cache_stats=manager.get_cache_stats(),
        )
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")