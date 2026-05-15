# ====================================================
# Emotion Lens Backend Dockerfile
# Multi-stage build for production deployment
# ====================================================

# ---- Base Stage ----
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies for PyTorch and ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies Stage ----
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Production Stage ----
FROM base AS production

# Copy Python dependencies
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ app/
COPY models/ models/
COPY alembic/ alembic/
COPY alembic.ini .

# Create data directories
RUN mkdir -p /data/feedback /data/monitoring /models/checkpoints

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu
ENV EMOTION_MODEL_PATH=/models/emotion_model
ENV FEEDBACK_DIR=/data/feedback
ENV MONITORING_DIR=/data/monitoring
ENV MODEL_SAVE_DIR=/models/checkpoints

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]