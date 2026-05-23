"""
Mental Health Analysis Routes
===============================
API endpoints for a CSV-trained meta-classifier combining sentiment, emotion
and symptom features.

Endpoints:
  POST /api/mental-health/analyze  — Analyze single text
  POST /api/mental-health/batch    — Analyze multiple texts
  GET  /api/mental-health/labels   — Get available labels
"""

import logging
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.models.mental_health_inference import (
    DETAILED_LABELS,
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
    all_scores: Dict[str, float] = Field(default_factory=dict)
    diagnosis_code: str = "normal"
    diagnosis_vi: str = "Không có dấu hiệu rõ ràng"
    diagnosis_scores: Dict[str, float] = Field(default_factory=dict)
    risk_level: str = "none"
    needs_attention: bool = False
    severity_level: int = 0
    severity_label: str = "healthy"
    top_predictions: List[TopPrediction] = Field(default_factory=list)
    extracted_features: Dict[str, float] = Field(default_factory=dict)
    disclaimer: str = "Ket qua chi ho tro sang loc, khong phai chan doan y khoa chinh thuc."
    dataset_loaded: bool = False
    validation_accuracy: Optional[float] = None
    source: str = "meta_rule_fallback"
    num_labels: int = 15
    processing_time_ms: float = 0.0


class BatchMentalHealthResponse(BaseModel):
    """Response model for batch mental health analysis."""
    results: List[MentalHealthResponse]
    total_processing_time_ms: float = 0.0


class LabelsResponse(BaseModel):
    """Response model for available labels."""
    num_labels: int = 15
    labels: List[str] = Field(default_factory=lambda: DETAILED_LABELS)
    ui_groups: List[str] = Field(default_factory=lambda: MENTAL_HEALTH_LABELS)
    description: str = "15-class mental health screening meta-classifier; UI groups remain backwards compatible"
    model: str = "RandomForest over sentiment, emotion and symptom features"


# ============================================================
# Routes
# ============================================================

@router.post("/analyze", response_model=MentalHealthResponse)
async def analyze_mental_health(request: MentalHealthRequest):
    """
    Analyze text for mental health conditions.
    
    Uses a meta-classifier trained from the configured rule-feature CSV. The
    response returns the detailed screening label plus a backwards-compatible
    grouped condition for the current UI.
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
            diagnosis_code=result.get("diagnosis_code", "normal"),
            diagnosis_vi=result.get("diagnosis_vi", "Không có dấu hiệu rõ ràng"),
            diagnosis_scores=result.get("diagnosis_scores", {}),
            risk_level=result.get("risk_level", "none"),
            needs_attention=result.get("needs_attention", False),
            severity_level=result.get("severity_level", 0),
            severity_label=result.get("severity_label", "healthy"),
            top_predictions=top_preds,
            extracted_features=result.get("extracted_features", {}),
            disclaimer=result.get("disclaimer", ""),
            dataset_loaded=result.get("dataset_loaded", False),
            validation_accuracy=result.get("validation_accuracy"),
            source=result.get("source", "meta_rule_fallback"),
            num_labels=result.get("num_labels", 15),
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
                diagnosis_code=result.get("diagnosis_code", "normal"),
                diagnosis_vi=result.get("diagnosis_vi", "Không có dấu hiệu rõ ràng"),
                diagnosis_scores=result.get("diagnosis_scores", {}),
                risk_level=result.get("risk_level", "none"),
                needs_attention=result.get("needs_attention", False),
                severity_level=result.get("severity_level", 0),
                severity_label=result.get("severity_label", "healthy"),
                top_predictions=top_preds,
                extracted_features=result.get("extracted_features", {}),
                disclaimer=result.get("disclaimer", ""),
                dataset_loaded=result.get("dataset_loaded", False),
                validation_accuracy=result.get("validation_accuracy"),
                source=result.get("source", "meta_rule_fallback"),
                num_labels=result.get("num_labels", 15),
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
    Get detailed CSV target labels and backwards-compatible UI groups.
    """
    return LabelsResponse()
