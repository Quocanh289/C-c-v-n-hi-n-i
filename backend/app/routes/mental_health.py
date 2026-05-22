"""
Mental Health Analysis Routes
===============================
API endpoints for mental health condition detection using the trained
DeBERTa-v3 + LoRA model.

Endpoints:
  POST /api/mental-health/analyze  — Analyze single text
  POST /api/mental-health/batch    — Analyze multiple texts
  GET  /api/mental-health/labels   — Get available labels
"""

import logging
import time
from typing import List, Optional, Any

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.models.mental_health_inference import (
    MentalHealthInference,
    get_mental_health_inference,
    MENTAL_HEALTH_LABELS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mental-health", tags=["mental_health"])


# ============================================================
# Pydantic Models
# ============================================================

class MentalHealthRequest(BaseModel):
    """Request model for mental health analysis."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to analyze")


class BatchMentalHealthRequest(BaseModel):
    """Request model for batch mental health analysis."""
    texts: List[str] = Field(..., min_length=1, max_length=50, description="List of texts to analyze")


class TopPrediction(BaseModel):
    """A single top prediction."""
    label: str
    confidence: float = 0.0


class MentalHealthResponse(BaseModel):
    """Response model for mental health analysis."""
    text: str
    primary_condition: str = "Normal"
    primary_confidence: float = 0.0
    all_scores: dict = {}
    needs_attention: bool = False
    severity_level: int = 0
    severity_label: str = "healthy"
    top_predictions: List[TopPrediction] = []
    source: str = "keyword_fallback"
    num_labels: int = 7
    processing_time_ms: float = 0.0


class BatchMentalHealthResponse(BaseModel):
    """Response model for batch mental health analysis."""
    results: List[MentalHealthResponse]
    total_processing_time_ms: float = 0.0


class LabelsResponse(BaseModel):
    """Response model for available labels."""
    num_labels: int = 7
    labels: List[str] = Field(default_factory=lambda: MENTAL_HEALTH_LABELS)
    description: str = "7-class mental health condition detection"
    model: str = "DeBERTa-v3-base + LoRA"


# ============================================================
# Routes
# ============================================================

@router.post("/analyze", response_model=MentalHealthResponse)
async def analyze_mental_health(request: MentalHealthRequest):
    """
    Analyze text for mental health conditions.
    
    Uses the trained 7-class DeBERTa-v3 + LoRA model to detect:
      - Normal (healthy)
      - Depression
      - Anxiety
      - Bipolar
      - Stress
      - Suicidal
      - Personality_disorder
    
    Returns severity assessment and top predictions.
    """
    start = time.time()
    
    try:
        infer = get_mental_health_inference()
        result = infer.classify(text=request.text)
        
        processing_time = (time.time() - start) * 1000
        
        # Build top predictions
        top_preds = [
            TopPrediction(label=p["label"], confidence=p["confidence"])
            for p in result.get("top_predictions", [])
        ]
        
        return MentalHealthResponse(
            text=request.text[:200] + "..." if len(request.text) > 200 else request.text,
            primary_condition=result.get("primary_condition", "Normal"),
            primary_confidence=result.get("primary_confidence", 0.0),
            all_scores=result.get("all_scores", {}),
            needs_attention=result.get("needs_attention", False),
            severity_level=result.get("severity_level", 0),
            severity_label=result.get("severity_label", "healthy"),
            top_predictions=top_preds,
            source=result.get("source", "keyword_fallback"),
            num_labels=result.get("num_labels", 7),
            processing_time_ms=processing_time,
        )
    except Exception as e:
        logger.error(f"Mental health analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mental health analysis failed: {str(e)}")


@router.post("/batch", response_model=BatchMentalHealthResponse)
async def analyze_mental_health_batch(request: BatchMentalHealthRequest):
    """
    Analyze multiple texts for mental health conditions.
    
    Each text is analyzed independently.
    Returns severity assessments and top predictions for each.
    """
    start = time.time()
    
    try:
        infer = get_mental_health_inference()
        results = []
        
        for text in request.texts:
            result = infer.classify(text=text)
            
            top_preds = [
                TopPrediction(label=p["label"], confidence=p["confidence"])
                for p in result.get("top_predictions", [])
            ]
            
            results.append(MentalHealthResponse(
                text=text[:200] + "..." if len(text) > 200 else text,
                primary_condition=result.get("primary_condition", "Normal"),
                primary_confidence=result.get("primary_confidence", 0.0),
                all_scores=result.get("all_scores", {}),
                needs_attention=result.get("needs_attention", False),
                severity_level=result.get("severity_level", 0),
                severity_label=result.get("severity_label", "healthy"),
                top_predictions=top_preds,
                source=result.get("source", "keyword_fallback"),
                num_labels=result.get("num_labels", 7),
                processing_time_ms=0.0,
            ))
        
        return BatchMentalHealthResponse(
            results=results,
            total_processing_time_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        logger.error(f"Batch mental health analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/labels", response_model=LabelsResponse)
async def get_mental_health_labels():
    """
    Get available mental health condition labels.
    Returns the 7-class label set with metadata.
    """
    return LabelsResponse()