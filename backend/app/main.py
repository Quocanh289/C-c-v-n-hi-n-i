# ====================================================
# Emotion Lens Backend API
# FastAPI application for emotion detection with
# PyTorch/HuggingFace models, Redis caching,
# Qdrant vector storage, and Celery task queue
# ====================================================

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.routes import analyze, learning, slang, health
from app.models.inference import GoEmotionsInference, get_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Global inference instance (lazy-loaded via get_inference)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle: pre-load model on startup."""
    global _inference_instance
    
    # Startup
    logger.info("Starting Emotion Lens Backend...")
    
    # Pre-load the GoEmotions 28-label model on startup
    try:
        infer = get_inference()
        if infer.is_loaded:
            logger.info("GoEmotions 28-label model loaded successfully on startup")
        else:
            logger.warning("GoEmotions model not available on startup (will load on first request)")
    except Exception as e:
        logger.warning(f"Could not pre-load model on startup: {e}")
        logger.info("Model will be loaded on first request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Emotion Lens Backend...")


# Create FastAPI app
app = FastAPI(
    title="Emotion Lens API",
    description="AI-powered emotion detection for social media text. "
                "Supports Vietnamese and English with multi-task learning.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - allow extension to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://*",  # Chrome extension
        "moz-extension://*",     # Firefox extension
        "http://localhost:3000", # Local development
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(analyze.router, prefix="/api", tags=["analysis"])
app.include_router(learning.router, prefix="/api", tags=["learning"])
app.include_router(slang.router, prefix="/api", tags=["slang"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Emotion Lens API",
        "version": "1.0.0",
        "description": "AI-powered social media emotion detection",
        "endpoints": {
            "health": "/api/health",
            "analyze": "/api/analyze",
            "batch_analyze": "/api/analyze/batch",
            "slang_detect": "/api/slang/detect",
            "slang_report": "/api/slang/report",
            "learning_feedback": "/api/learning/feedback",
            "learning_retrain": "/api/learning/retrain",
        },
    }