FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright browsers for TikTok/Facebook/Threads
RUN pip install playwright && playwright install chromium --with-deps

COPY . /app/data_engine
ENV PYTHONPATH=/app

EXPOSE 8100

CMD ["uvicorn", "data_engine.api.main:app", "--host", "0.0.0.0", "--port", "8100"]
