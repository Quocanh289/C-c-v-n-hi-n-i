# ====================================================
# Health Check Route
# ====================================================

import os
import time
import logging
from datetime import datetime

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

start_time = time.time()


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    
    uptime_seconds = time.time() - start_time
    
    # Check Redis connection if configured
    redis_status = "not_configured"
    redis_host = os.getenv("REDIS_HOST", "")
    if redis_host:
        try:
            import redis.asyncio as aioredis
            r = await aioredis.from_url(f"redis://{redis_host}:{os.getenv('REDIS_PORT', '6379')}")
            await r.ping()
            redis_status = "connected"
            await r.close()
        except Exception as e:
            redis_status = f"error: {str(e)[:50]}"
    
    return {
        "status": "healthy",
        "service": "emotion-lens-api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime_seconds,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "dependencies": {
            "redis": redis_status,
        },
    }