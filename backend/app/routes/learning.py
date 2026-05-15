# ====================================================
# Continuous Learning Routes
# Handles feedback collection, retraining triggers,
# and model versioning for incremental learning
# ====================================================

import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, BackgroundTasks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learning", tags=["learning"])


# ====================================================
# Pydantic Models
# ====================================================

class FeedbackItem(BaseModel):
    """A single feedback item for continuous learning."""
    text: str = Field(..., description="The original text")
    predicted_emotion: str = Field(..., description="Emotion predicted by the model")
    corrected_emotion: str = Field(..., description="Correct emotion label (human-verified)")
    predicted_toxicity: Optional[float] = None
    corrected_toxicity: Optional[float] = None
    predicted_sarcasm: Optional[float] = None
    corrected_sarcasm: Optional[float] = None
    confidence: float = Field(..., ge=0, le=1)
    language: str = "en"
    source: str = "extension"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class FeedbackRequest(BaseModel):
    """Request to submit feedback for model improvement."""
    feedbacks: List[FeedbackItem] = Field(..., min_length=1, max_length=100)


class RetrainRequest(BaseModel):
    """Request to trigger model retraining."""
    epochs: int = Field(3, ge=1, le=20)
    learning_rate: float = Field(2e-5, ge=1e-6, le=1e-3)
    batch_size: int = Field(16, ge=4, le=128)
    validation_split: float = Field(0.1, ge=0.0, le=0.3)
    use_lora: bool = True


class ModelVersionInfo(BaseModel):
    """Information about a model version."""
    version: str
    created_at: str
    metrics: Dict = {}
    status: str  # "active" | "staging" | "archived"
    path: str


# ====================================================
# Feedback Storage
# ====================================================

FEEDBACK_DIR = os.getenv("FEEDBACK_DIR", "data/feedback")
os.makedirs(FEEDBACK_DIR, exist_ok=True)


def _save_feedback(feedback: FeedbackItem):
    """Save individual feedback to disk (JSONL format)."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    filepath = os.path.join(FEEDBACK_DIR, f"feedback_{date_str}.jsonl")
    
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback.dict(), ensure_ascii=False) + "\n")


# ====================================================
# Routes
# ====================================================

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit feedback for continuous model improvement.
    
    User-corrected labels are saved and used for:
    1. Online evaluation metrics
    2. Periodic retraining datasets
    3. Slang detection refinement
    
    Feedback is processed asynchronously to avoid blocking.
    """
    try:
        # Save feedback in background
        for feedback in request.feedbacks:
            background_tasks.add_task(_save_feedback, feedback)
        
        # Calculate basic stats
        corrections = sum(
            1 for f in request.feedbacks
            if f.predicted_emotion != f.corrected_emotion
        )
        
        return {
            "status": "accepted",
            "total_items": len(request.feedbacks),
            "corrections": corrections,
            "accuracy": 1.0 - (corrections / len(request.feedbacks)) if request.feedbacks else 1.0,
            "message": f"Feedback saved. {corrections} corrections out of {len(request.feedbacks)} items.",
        }
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get statistics about collected feedback.
    """
    try:
        total_files = 0
        total_items = 0
        corrections = 0
        
        for filename in os.listdir(FEEDBACK_DIR):
            if filename.startswith("feedback_") and filename.endswith(".jsonl"):
                total_files += 1
                filepath = os.path.join(FEEDBACK_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        total_items += 1
                        if item.get("predicted_emotion") != item.get("corrected_emotion"):
                            corrections += 1
        
        return {
            "total_days": total_files,
            "total_feedback_items": total_items,
            "total_corrections": corrections,
            "overall_accuracy": 1.0 - (corrections / total_items) if total_items > 0 else 1.0,
            "feedback_directory": FEEDBACK_DIR,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
async def trigger_retraining(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger model retraining with accumulated feedback.
    
    Uses LoRA fine-tuning for efficient retraining:
    - Only trains adapter weights (not full model)
    - Preserves base model knowledge
    - Fast convergence (3-5 epochs)
    
    The retraining runs asynchronously in the background.
    """
    try:
        # Import training utilities
        from app.models.training import train_emotion_model
        
        # Background task for retraining
        background_tasks.add_task(
            train_emotion_model,
            feedback_dir=FEEDBACK_DIR,
            model_save_dir=os.getenv("MODEL_SAVE_DIR", "models/checkpoints"),
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            validation_split=request.validation_split,
            use_lora=request.use_lora,
        )
        
        return {
            "status": "retraining_started",
            "config": request.dict(),
            "message": "Retraining initiated. This runs asynchronously in the background.",
            "note": "Use GET /api/learning/versions to check training status.",
        }
    except Exception as e:
        logger.error(f"Failed to start retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")


@router.get("/versions", response_model=List[ModelVersionInfo])
async def list_model_versions():
    """
    List all available model versions.
    """
    model_dir = os.getenv("MODEL_SAVE_DIR", "models/checkpoints")
    versions = []
    
    if not os.path.exists(model_dir):
        return versions
    
    try:
        for version_dir in sorted(os.listdir(model_dir), reverse=True):
            version_path = os.path.join(model_dir, version_dir)
            if os.path.isdir(version_path):
                config_path = os.path.join(version_path, "config.json")
                metrics_path = os.path.join(version_path, "metrics.json")
                
                config = {}
                metrics = {}
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        config = json.load(f)
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                
                versions.append(ModelVersionInfo(
                    version=version_dir,
                    created_at=config.get("created_at", "unknown"),
                    metrics=metrics,
                    status=config.get("status", "staging"),
                    path=version_path,
                ))
        
        return versions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))