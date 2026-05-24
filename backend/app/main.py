# ====================================================
# Emotion Lens Backend API
# FastAPI application for emotion detection with
# PyTorch/HuggingFace models, Redis caching,
# Qdrant vector storage, and Celery task queue
# ====================================================

import logging

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.routes import analyze, learning, slang, health, mental_health
from app.routes.dashboard import router as dashboard_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle: pre-load model on startup."""
    # Startup
    logger.info("Starting Emotion Lens Backend...")
    
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
app.include_router(dashboard_router)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware - allow extension to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", # Local development
        "http://localhost:5173",
    ],
    allow_origin_regex=r"^(chrome-extension|moz-extension)://.*$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(analyze.router, prefix="/api", tags=["analysis"])
app.include_router(learning.router, prefix="/api", tags=["learning"])
app.include_router(slang.router, prefix="/api", tags=["slang"])
app.include_router(mental_health.router, prefix="/api", tags=["mental_health"])


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
            "mental_health_analyze": "/api/mental-health/analyze",
            "mental_health_batch": "/api/mental-health/batch",
            "mental_health_labels": "/api/mental-health/labels",
        },
    }
