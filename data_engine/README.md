# Emotion Lens — Data Engine

Scalable Vietnamese/English social-media crawling and dataset generation for NLP emotion detection.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for full system design.

## Features

- **6 platforms**: Reddit, YouTube, TikTok, Facebook (public), Threads, X/Twitter
- **Distributed crawling**: Celery workers + Redis queues
- **3-layer dedup**: exact hash, MinHash, semantic (Qdrant)
- **Emotion-preserving NLP pipeline**: keeps emojis, slang, repeated chars
- **Slang drift detection**: emerging term tracking + human review
- **Labeling**: active learning + Label Studio + consensus scoring
- **Export**: JSON, JSONL, Hugging Face datasets for XLM-RoBERTa / LoRA

## Setup

```bash
cd data_engine
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

## Run locally

From the **repo root** (`C-c-v-n-hi-n-i`), not from inside `data_engine/scripts/`:

```bash
# Optional: install package in editable mode (fixes imports everywhere)
pip install -e .

# Single crawl (works from repo root or scripts/)
python data_engine/scripts/run_crawl.py --platform reddit --target vietnam --limit 50

# Or as a module
python -m data_engine.scripts.run_crawl --platform reddit --target vietnam --limit 50

# API
uvicorn data_engine.api.main:app --reload --port 8100

# Celery worker
celery -A data_engine.workers.celery_app worker -Q crawl,process,analytics,export -l info

# Celery beat
celery -A data_engine.workers.celery_app beat -l info
```

## Docker

```bash
docker compose -f docker-compose.yml -f data_engine/docker-compose.data-engine.yml up -d
```

## Crawl → Clean → Label workflow

```bash
# 1. Crawl + clean + save to disk (one command)
python data_engine/scripts/run_crawl.py \
  --platform youtube \
  --target "https://www.youtube.com/watch?v=Uui8-K5kCLY" \
  --limit 100 \
  --save

# Output:
#   data/crawls/raw/youtube_YYYYMMDD_HHMMSS_raw.jsonl      — raw crawler output
#   data/crawls/clean/youtube_..._clean.jsonl              — after NLP pipeline
#   data/crawls/labeling/youtube_..._for_labeling.jsonl    — import to Label Studio

# 2. Re-clean an existing raw file
python data_engine/scripts/process_raw_file.py data/crawls/raw/youtube_*_raw.jsonl --platform youtube

# 3. Optional: PostgreSQL (docker compose up -d postgres)
python data_engine/scripts/run_crawl.py ... --save --save-db
```

Each **labeling** line looks like:

```json
{
  "text": "comment gốc giữ emoji 😭",
  "text_clean": "đã chuẩn hoá nhẹ",
  "language": "vi",
  "source": "youtube",
  "emotion": null,
  "toxicity": null,
  "sarcasm": null
}
```

Fill `emotion` / `toxicity` / `sarcasm` via Label Studio or your annotation tool, then export for training (`ai_nlp/training/emotion_pipeline/`).

## API Examples

```bash
# Start Reddit crawl
curl -X POST http://localhost:8100/api/v1/crawl \
  -H "Content-Type: application/json" \
  -d '{"platform": "reddit", "target": "vietnam"}'

# Trigger slang drift scan
curl -X POST http://localhost:8100/api/v1/slang/drift

# Export dataset
curl -X POST http://localhost:8100/api/v1/dataset/export \
  -H "Content-Type: application/json" \
  -d '{"format": "jsonl"}'
```
