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
import re
import sys
import time
import torch
import unicodedata
from typing import Dict, List, Optional
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.mental_health_inference import DETAILED_LABELS
from app.routes.translate import contains_vietnamese, translate_simple

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


MENTAL_HEALTH_SIGNAL_PATTERNS = {
    "Suicidal": [
        "kill myself", "end my life", "want to die", "suicide", "self harm",
        "don't want to live", "dont want to live", "better off dead",
        "muon chet", "tu tu", "tu sat", "khong muon song", "tu lam hai",
    ],
    "Depression": [
        "depressed", "depression", "hopeless", "worthless", "empty", "numb",
        "no motivation", "lost interest", "can't get out of bed", "cant get out of bed",
        "i hate myself", "meaningless", "sad all the time", "crying everyday",
        "tram cam", "tuyet vong", "vo dung", "vo gia tri", "mat hung thu",
        "khong con hung thu", "rat buon", "buon ca ngay", "khoc moi ngay",
    ],
    "Anxiety": [
        "anxious", "anxiety", "panic", "panic attack", "overthinking",
        "can't stop worrying", "cant stop worrying", "constant worry", "heart racing",
        "can't breathe", "cant breathe", "social anxiety",
        "lo lang", "hoang loan", "bon chon", "tim dap nhanh", "kho tho",
        "so hai", "so giao tiep", "suy nghi qua nhieu",
    ],
    "Bipolar": [
        "manic", "mania", "hypomania", "hypomanic", "bipolar", "mood swings",
        "rapid cycling", "grandiose", "pressured speech", "flight of ideas",
        "hung cam", "luong cuc", "dao dong khi sac", "luc vui luc buon",
    ],
    "Stress": [
        "stressed", "stress", "overwhelmed", "burnout", "burned out",
        "can't cope", "cant cope", "can't handle it", "cant handle it",
        "too much pressure", "overworked", "exhausted", "insomnia",
        "can't sleep", "cant sleep", "cannot sleep",
        "cang thang", "qua tai", "ap luc", "kiet suc", "met moi", "khong chiu noi",
        "khong the doi pho", "mat ngu", "khong ngu",
    ],
    "Personality_disorder": [
        "bpd", "borderline", "personality disorder", "abandonment issues",
        "unstable relationships", "identity disturbance", "emotional dysregulation",
        "narcissistic", "dissociative", "depersonalization", "derealization",
        "roi loan nhan cach", "so bi bo roi", "mat ket noi thuc tai",
    ],
}

MENTAL_HEALTH_CONTEXT_TERMS = [
    "mental health", "therapy", "therapist", "psychiatrist", "psychologist",
    "diagnosed", "diagnosis", "medication", "antidepressant", "ssri",
    "suc khoe tam than", "tri lieu", "bac si tam ly", "chan doan", "thuoc tram cam",
]

SIGNAL_LABEL_PRIORITY = {
    "Suicidal": 6,
    "Depression": 5,
    "Anxiety": 4,
    "Stress": 3,
    "Bipolar": 2,
    "Personality_disorder": 1,
}

MIN_CONFIDENCE_FOR_UNMATCHED_LABEL = 0.35


def _normalize_for_gate(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "d"))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = re.sub(r"[^a-z0-9\s']", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _has_mental_health_signal(text: str) -> bool:
    normalized = _normalize_for_gate(text)
    if not normalized:
        return False
    if any(term in normalized for term in MENTAL_HEALTH_CONTEXT_TERMS):
        return True
    return any(
        pattern in normalized
        for patterns in MENTAL_HEALTH_SIGNAL_PATTERNS.values()
        for pattern in patterns
    )


def _matched_signal_labels(*texts: Optional[str]) -> set[str]:
    labels: set[str] = set()
    for text in texts:
        if not text:
            continue
        normalized = _normalize_for_gate(text)
        for label, patterns in MENTAL_HEALTH_SIGNAL_PATTERNS.items():
            if any(pattern in normalized for pattern in patterns):
                labels.add(label)
    return labels


def _normal_result(text: str, source: str = "domain_guard_normal") -> Dict[str, Any]:
    scores = {label: 0.0 for label in MENTAL_HEALTH_LABELS}
    scores["Normal"] = 1.0
    return {
        "text": text[:200],
        "primary_condition": "Normal",
        "primary_confidence": 1.0,
        "all_scores": scores,
        "top_predictions": [{"label": "Normal", "confidence": 1.0}],
        "risk_signals": [],
        "needs_attention": False,
        "severity_level": 0,
        "severity_label": MH_SEVERITY_LABELS.get(0, "healthy"),
        "source": source,
        "num_labels": len(MENTAL_HEALTH_LABELS),
    }


def _single_label_result(text: str, label: str, confidence: float, source: str) -> Dict[str, Any]:
    scores = {name: 0.0 for name in MENTAL_HEALTH_LABELS}
    scores[label] = confidence
    severity_level = MH_SEVERITY_MAP.get(label, 0)
    return {
        "text": text[:200],
        "primary_condition": label,
        "primary_confidence": confidence,
        "all_scores": scores,
        "top_predictions": [{"label": label, "confidence": confidence}],
        "risk_signals": [] if label == "Normal" else [{"label": label, "score": confidence}],
        "needs_attention": label != "Normal",
        "severity_level": severity_level,
        "severity_label": MH_SEVERITY_LABELS.get(severity_level, "healthy"),
        "source": source,
        "num_labels": len(MENTAL_HEALTH_LABELS),
    }


def _best_matched_label(labels: set[str]) -> Optional[str]:
    candidates = [label for label in labels if label in SIGNAL_LABEL_PRIORITY]
    if not candidates:
        return None
    return max(candidates, key=lambda label: SIGNAL_LABEL_PRIORITY[label])


def _risk_signals_from_scores(scores: Dict[str, float], min_score: float = 0.15) -> List[Dict[str, float | str]]:
    return [
        {"label": label, "score": round(float(score), 4)}
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if label != "Normal" and score >= min_score
    ][:4]


def _risk_signals_from_matches(labels: set[str], score: float = 0.5) -> List[Dict[str, float | str]]:
    return [
        {"label": label, "score": score}
        for label in sorted(labels, key=lambda item: SIGNAL_LABEL_PRIORITY.get(item, 0), reverse=True)
        if label != "Normal"
    ][:4]


class MentalHealthInference:
    """Lazy mental health inference adapter backed by ai_nlp."""

    def __init__(self) -> None:
        self.config = TrainingConfig()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._load_model_if_available()

    # Prefer the promoted best_model requested by the UI, then session_best, then outputs.
    CANDIDATE_DIRS = [
        os.path.join(str(Path(__file__).resolve().parents[3]), "ai_nlp", "training", "checkpoints", "mental_health_model", "best_model"),
        os.path.join(str(Path(__file__).resolve().parents[3]), "ai_nlp", "training", "checkpoints", "mental_health_model", "session_best_20260523_205703"),
        os.path.join(str(Path(__file__).resolve().parents[3]), "ai_nlp", "training", "outputs", "mental_health_model"),
    ]

    def _load_model_if_available(self) -> None:
        # Find first valid model directory
        model_dir = None
        for candidate in self.CANDIDATE_DIRS:
            if os.path.exists(os.path.join(candidate, "adapter_model.safetensors")):
                model_dir = candidate
                break

        if model_dir is None:
            logger.warning("No trained mental health model found; using keyword fallback.")
            return

        if AutoTokenizer is None:
            logger.warning("transformers is unavailable; using keyword fallback.")
            return

        try:
            import json
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer as HFAutoTokenizer
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading mental health model from %s on device: %s", model_dir, device)
            
            # Load base DeBERTa + LoRA adapter directly (no MentalHealthClassifier wrapper)
            model_config = AutoConfig.from_pretrained(
                "microsoft/deberta-v3-base", num_labels=7
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-v3-base", config=model_config,
                torch_dtype=torch.float32, ignore_mismatched_sizes=True,
            ).to(device)
            
            self.model = PeftModel.from_pretrained(base_model, model_dir)
            self.model = self.model.to(device)
            self.model.eval()
            
            self.tokenizer = HFAutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
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
        if not _has_mental_health_signal(text):
            return _normal_result(text, "keyword_fallback_domain_guard")

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
            "risk_signals": _risk_signals_from_scores(scores, min_score=0.10),
            "needs_attention": needs_attention,
            "severity_level": severity_level,
            "severity_label": MH_SEVERITY_LABELS.get(severity_level, "healthy"),
            "source": "keyword_fallback",
            "num_labels": len(MENTAL_HEALTH_LABELS),
        }

    def classify(self, text: str, signal_text: Optional[str] = None) -> Dict[str, Any]:
        matched_labels = _matched_signal_labels(text, signal_text)
        has_signal = bool(matched_labels) or _has_mental_health_signal(text) or (
            signal_text is not None and _has_mental_health_signal(signal_text)
        )
        if not has_signal:
            return _normal_result(text)
        if "Suicidal" in matched_labels:
            return _single_label_result(text, "Suicidal", 1.0, "safety_rule_override")

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
        if (
            primary_condition != "Normal"
            and primary_condition not in matched_labels
            and confidence_val < MIN_CONFIDENCE_FOR_UNMATCHED_LABEL
        ):
            corrected_label = _best_matched_label(matched_labels)
            if corrected_label:
                return _single_label_result(
                    text,
                    corrected_label,
                    max(0.5, float(confidence_val)),
                    "signal_rule_correction",
                )

        severity_level = MH_SEVERITY_MAP.get(primary_condition, 0)

        scores = {
            label: float(probs_np[i])
            for i, label in enumerate(MENTAL_HEALTH_LABELS)
        }
        risk_signals = _risk_signals_from_matches(matched_labels)
        if not risk_signals:
            risk_signals = _risk_signals_from_scores(scores)

        return {
            "text": text[:200],
            "primary_condition": primary_condition,
            "primary_confidence": confidence_val,
            "all_scores": scores,
            "top_predictions": self._build_top_predictions(scores),
            "risk_signals": risk_signals,
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


class RiskSignal(BaseModel):
    """A non-diagnostic screening signal surfaced for UI display."""
    label: str
    score: float = 0.0


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
    risk_signals: List[RiskSignal] = Field(default_factory=list)
    extracted_features: Dict[str, float] = Field(default_factory=dict)
    disclaimer: str = "Ket qua chi ho tro sang loc, khong phai chan doan y khoa chinh thuc."
    dataset_loaded: bool = False
    validation_accuracy: Optional[float] = None
    source: str = "meta_rule_fallback"
    num_labels: int = 15
    processing_time_ms: float = 0.0
    language: str = "en"
    translation: Dict[str, str] = Field(default_factory=dict)


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
        text_for_analysis = request.text
        translation: Dict[str, str] = {}
        if contains_vietnamese(request.text):
            text_for_analysis = translate_simple(request.text)
            translation = {
                "source_language": "vi",
                "target_language": "en",
                "translated_text": text_for_analysis,
            }
            logger.info("Translated VI->EN for mental health analysis: %s -> %s", request.text[:60], text_for_analysis[:60])

        infer = get_mental_health_inference()
        result = infer.classify(text=text_for_analysis, signal_text=request.text)
        
        processing_time = (time.time() - start) * 1000
        
        # Build top predictions
        top_preds = [
            TopPrediction(label=p["label"], confidence=p["confidence"])
            for p in result.get("top_predictions", [])
        ]
        risk_signals = [
            RiskSignal(label=s["label"], score=s["score"])
            for s in result.get("risk_signals", [])
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
            risk_signals=risk_signals,
            extracted_features=result.get("extracted_features", {}),
            disclaimer=result.get("disclaimer", ""),
            dataset_loaded=result.get("dataset_loaded", False),
            validation_accuracy=result.get("validation_accuracy"),
            source=result.get("source", "meta_rule_fallback"),
            num_labels=result.get("num_labels", 15),
            processing_time_ms=processing_time,
            language="vi" if translation else "en",
            translation=translation,
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
            text_for_analysis = translate_simple(text) if contains_vietnamese(text) else text
            translation = {
                "source_language": "vi",
                "target_language": "en",
                "translated_text": text_for_analysis,
            } if text_for_analysis != text else {}
            result = infer.classify(text=text_for_analysis, signal_text=text)
            
            top_preds = [
                TopPrediction(label=p["label"], confidence=p["confidence"])
                for p in result.get("top_predictions", [])
            ]
            risk_signals = [
                RiskSignal(label=s["label"], score=s["score"])
                for s in result.get("risk_signals", [])
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
                risk_signals=risk_signals,
                extracted_features=result.get("extracted_features", {}),
                disclaimer=result.get("disclaimer", ""),
                dataset_loaded=result.get("dataset_loaded", False),
                validation_accuracy=result.get("validation_accuracy"),
                source=result.get("source", "meta_rule_fallback"),
                num_labels=result.get("num_labels", 15),
                processing_time_ms=0.0,
                language="vi" if translation else "en",
                translation=translation,
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
