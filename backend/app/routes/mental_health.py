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
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ai_nlp.training.mental_health_pipeline.config import (
    MH_KEYWORDS,
    MH_SEVERITY_LABELS,
    MH_SEVERITY_MAP,
    MENTAL_HEALTH_LABELS,
    TrainingConfig,
)
from ai_nlp.training.mental_health_pipeline.model import load_trained_model

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - transformers is already a backend dependency
    AutoTokenizer = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mental-health", tags=["mental_health"])


class MentalHealthInference:
    """Lazy mental health inference adapter backed by ai_nlp."""

    def __init__(self) -> None:
        self.config = TrainingConfig()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        model_dir = self.config.get_output_model_dir()
        adapter_path = os.path.join(model_dir, "adapter_model.safetensors")
        model_bin_path = os.path.join(model_dir, "pytorch_model.bin")

        if not os.path.exists(adapter_path) and not os.path.exists(model_bin_path):
            logger.warning(
                "No trained mental health model found at %s; using keyword fallback.",
                model_dir,
            )
            return

        if AutoTokenizer is None:
            logger.warning("transformers is unavailable; using keyword fallback.")
            return

        try:
            self.model = load_trained_model(model_dir, self.config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.is_loaded = True
            logger.info("Loaded mental health model from %s", model_dir)
        except Exception as exc:
            logger.warning(
                "Could not load trained mental health model from %s; falling back to keywords: %s",
                model_dir,
                exc,
            )
            self.model = None
            self.tokenizer = None
            self.is_loaded = False

    def _build_top_predictions(self, scores: Dict[str, float]) -> List[Dict[str, float]]:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
        return [{"label": label, "confidence": float(confidence)} for label, confidence in ranked]

    def _keyword_fallback(self, text: str) -> Dict[str, Any]:
        lowered = text.lower()
        scores = {label: 0.0 for label in MENTAL_HEALTH_LABELS}

        for label, keywords in MH_KEYWORDS.items():
            if label not in scores:
                continue

            matched = sum(1 for keyword in keywords if keyword in lowered)
            if matched > 0:
                scores[label] = min(1.0, matched / max(len(keywords), 1))

        primary_condition = max(scores, key=scores.get)
        primary_confidence = float(scores[primary_condition])
        severity_level = MH_SEVERITY_MAP.get(primary_condition, 0)

        return {
            "text": text[:200],
            "primary_condition": primary_condition,
            "primary_confidence": primary_confidence,
            "all_scores": scores,
            "top_predictions": self._build_top_predictions(scores),
            "needs_attention": primary_condition != "Normal" and primary_confidence >= 0.3,
            "severity_level": severity_level,
            "severity_label": MH_SEVERITY_LABELS.get(severity_level, "healthy"),
            "source": "keyword_fallback",
            "num_labels": len(MENTAL_HEALTH_LABELS),
        }

    def classify(self, text: str) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            return self._keyword_fallback(text)

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)

        predictions, confidences = self.model.predict(input_ids, attention_mask)
        probs = self.model.get_probs(input_ids, attention_mask).detach().cpu().numpy()[0]

        prediction_index = int(predictions.item() if hasattr(predictions, "item") else predictions[0])
        confidence = float(confidences.item() if hasattr(confidences, "item") else confidences[0])
        primary_condition = MENTAL_HEALTH_LABELS[prediction_index]
        severity_level = MH_SEVERITY_MAP.get(primary_condition, 0)

        scores = {
            label: float(probs[i])
            for i, label in enumerate(MENTAL_HEALTH_LABELS)
        }

        return {
            "text": text[:200],
            "primary_condition": primary_condition,
            "primary_confidence": confidence,
            "all_scores": scores,
            "top_predictions": self._build_top_predictions(scores),
            "needs_attention": primary_condition != "Normal" and confidence >= 0.3,
            "severity_level": severity_level,
            "severity_label": MH_SEVERITY_LABELS.get(severity_level, "healthy"),
            "source": "trained_model",
            "num_labels": len(MENTAL_HEALTH_LABELS),
        }


@lru_cache(maxsize=1)
def get_mental_health_inference() -> MentalHealthInference:
    return MentalHealthInference()


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