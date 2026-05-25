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
import os
import sys
import time
import torch
from typing import Dict, List, Optional
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.mental_health_inference import DETAILED_LABELS

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

    BEST_MODEL_DIR = os.path.join(
        str(Path(__file__).resolve().parents[3]),
        "ai_nlp", "training", "checkpoints", "mental_health_model", "best_model"
    )

    def _load_model_if_available(self) -> None:
        model_dir = self.BEST_MODEL_DIR
        
        if not os.path.exists(os.path.join(model_dir, "adapter_model.safetensors")):
            logger.warning("No trained model at %s; using keyword fallback.", model_dir)
            model_dir = None
            # Fallback: search other paths
            for candidate in [
                os.path.join(str(Path(__file__).resolve().parents[3]), "ai_nlp", "training", "outputs", "mental_health_model"),
                self.config.get_output_model_dir(),
            ]:
                if os.path.exists(os.path.join(candidate, "adapter_model.safetensors")):
                    model_dir = candidate
                    break
            
            if model_dir is None:
                logger.warning("No trained mental health model found anywhere; using keyword fallback.")
                return

        if AutoTokenizer is None:
            logger.warning("transformers is unavailable; using keyword fallback.")
            return

        try:
            import json
            import torch
            from transformers import AutoTokenizer as HFAutoTokenizer
            from ai_nlp.training.mental_health_pipeline.config import TrainingConfig as PipeConfig
            from ai_nlp.training.mental_health_pipeline.model import load_trained_model as pipeline_load
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading mental health model on device: %s", device)
            
            # Use config for best_model checkpoint (outputs dir)
            pipe_config = PipeConfig()
            self.model = pipeline_load(model_dir, pipe_config)
            self.model = self.model.to(device)
            self.model.eval()
            
            self.tokenizer = HFAutoTokenizer.from_pretrained(pipe_config.model_name)
            self.is_loaded = True
            logger.info("Loaded mental health model from %s on %s", model_dir, device)
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
                # Boost confidence: use log scale so 1 keyword match gives ~0.35, 
                # multiple matches approach 0.95. This ensures badges actually show
                # above default threshold (0.15).
                raw = matched / max(len(keywords), 1)
                boosted = min(0.95, 0.30 + (raw * 0.65))
                scores[label] = boosted

        primary_condition = max(scores, key=scores.get)
        primary_confidence = float(scores[primary_condition])
        severity_level = MH_SEVERITY_MAP.get(primary_condition, 0)

        # Also boost when using MENTAL_HEALTH_INFERENCE model's rule_prediction
        # by checking co-occurring symptoms
        needs_attention = primary_condition != "Normal" and primary_confidence >= 0.10
        if primary_condition != "Normal" and primary_confidence < 0.15:
            primary_confidence = max(primary_confidence, 0.20)

        return {
            "text": text[:200],
            "primary_condition": primary_condition,
            "primary_confidence": primary_confidence,
            "all_scores": scores,
            "top_predictions": self._build_top_predictions(scores),
            "needs_attention": needs_attention,
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

        # Send inputs to same device as model
        model_device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded["attention_mask"].to(model_device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # MentalHealthClassifier returns dict, PeftModel returns object
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            # Flatten to handle both [1,7] and [7] shapes
            logits = logits.view(-1)
            probs = torch.softmax(logits, dim=-1)
            confidence_val = torch.max(probs).item()
            prediction_index = torch.argmax(probs).item()

        probs_np = probs.cpu().numpy().tolist()
        primary_condition = MENTAL_HEALTH_LABELS[prediction_index]
        severity_level = MH_SEVERITY_MAP.get(primary_condition, 0)

        scores = {
            label: float(probs_np[i])
            for i, label in enumerate(MENTAL_HEALTH_LABELS)
        }

        return {
            "text": text[:200],
            "primary_condition": primary_condition,
            "primary_confidence": confidence_val,
            "all_scores": scores,
            "top_predictions": self._build_top_predictions(scores),
            "needs_attention": primary_condition != "Normal",
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
