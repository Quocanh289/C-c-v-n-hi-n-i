# Data Engine — Architecture

Production-grade Vietnamese/English social-media crawling and NLP dataset generation for Emotion Lens.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTROL PLANE (FastAPI :8100)                        │
│  POST /api/v1/crawl  │  POST /api/v1/slang/drift  │  POST /dataset/export   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ Celery enqueue
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                    DISTRIBUTED WORKERS (Celery + Redis)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ crawl queue │  │ process queue│  │ analytics   │  │ export queue     │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘  │
└─────────┼────────────────┼─────────────────┼───────────────────┼────────────┘
          │                │                 │                   │
┌─────────▼────────────────▼─────────────────▼───────────────────▼────────────┐
│                         CRAWLER LAYER (modular)                              │
│  Reddit (API) │ YouTube (API) │ Twitter (API) │ TikTok/FB/Threads (PW)      │
│  BaseCrawler → RateLimiter → Retry → Dedup → RawPost                         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                      PROCESSING PIPELINE (preserve signals)                  │
│  Unicode NFC │ URL strip │ Spam filter │ Lang detect │ Emoji/slang KEEP       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐        ┌─────────▼─────────┐       ┌────────▼────────┐
│ MongoDB (raw)  │        │ PostgreSQL (dataset)│      │ Qdrant (vectors) │
│ raw_posts      │        │ dataset_records     │      │ semantic dedup   │
│ token_snapshots│        │ slang_candidates    │      │ slang clusters   │
└────────────────┘        └─────────────────────┘       └─────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                    LABELING & EXPORT                                         │
│  Active Learning → Label Studio → Consensus → JSON/JSONL/HuggingFace         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Folder Structure

```
data_engine/
├── ARCHITECTURE.md          # This document
├── README.md                # Quick start
├── requirements.txt
├── Dockerfile
├── docker-compose.data-engine.yml
├── .env.example
├── config/
│   ├── settings.py          # Pydantic settings
│   └── platforms.yaml       # Per-platform rate limits & schedules
├── core/
│   ├── base_crawler.py      # Abstract crawler contract
│   ├── browser_session.py   # Playwright pool + infinite scroll
│   ├── queue.py             # Redis priority queue
│   ├── rate_limiter.py      # Distributed RPM limiter
│   ├── retry.py             # Tenacity exponential backoff
│   ├── dedup.py             # SHA256 + MinHash LSH
│   ├── scheduler.py         # Cron-based job builder
│   └── models.py            # Pydantic domain models
├── crawlers/
│   ├── reddit.py            # OAuth API + public JSON fallback
│   ├── youtube.py           # Data API v3 commentThreads
│   ├── tiktok.py            # Playwright + infinite scroll
│   ├── facebook.py          # Public pages only
│   ├── threads.py           # Playwright
│   └── twitter.py           # API v2 search
├── processing/
│   ├── normalizer.py        # Light clean — preserves emojis/slang
│   ├── language.py          # vi / en / mixed detection
│   ├── spam_filter.py       # Conservative spam rules
│   └── pipeline.py          # RawPost → ProcessedRecord
├── slang/
│   └── drift_detector.py    # Token frequency growth detection
├── embeddings/
│   ├── encoder.py           # multilingual-MiniLM
│   └── qdrant_store.py      # Semantic dedup + clustering
├── labeling/
│   ├── active_learning.py   # Uncertainty sampling
│   └── label_studio.py      # Multi-annotator integration
├── dataset/
│   ├── exporter.py          # JSON / JSONL export
│   └── huggingface_integration.py
├── storage/
│   ├── postgres_repo.py     # Structured dataset + slang
│   └── mongo_store.py       # Raw documents + snapshots
├── workers/
│   ├── celery_app.py        # Queue routing + beat schedule
│   └── tasks.py             # crawl / drift / export tasks
├── api/
│   └── main.py              # FastAPI control plane
└── database/
    └── init_data_engine.sql # PG schema extension
```

## Deduplication (3 Layers)

| Layer | Mechanism | Storage | Purpose |
|-------|-----------|---------|---------|
| L1 Exact | SHA256 normalized hash | Redis SET | Identical text |
| L2 Near | MinHash Jaccard ≥ 0.85 | Redis LSH buckets | Paraphrase / copypasta |
| L3 Semantic | Cosine ≥ 0.92 | Qdrant | Meaning-level duplicates |

## NLP Preprocessing Philosophy

**Do NOT over-clean.** Training data uses `text_raw` for model input; `text` is lightly normalized for indexing only.

Preserved signals:
- Emojis (`😭`, `💀`)
- Repeated characters (`đỉnhhh`, `soooo`)
- Internet slang (`cooked`, `delulu`, `xỉu`)
- Expressive punctuation (`!!!`, `???`)

Removed only:
- URLs, @mentions
- Excessive whitespace
- Extreme char repetition (8+ → 4)

## Slang Drift Detection

1. Snapshot token frequencies per 7-day window (MongoDB `token_snapshots`)
2. Compare current vs previous window
3. Flag tokens with growth rate ≥ 3× and frequency ≥ 5
4. Store in `slang_candidates` for human review
5. Cluster similar candidates via Qdrant embeddings

Example emerging terms: `cooked`, `delulu`, `xỉu up xỉu down`, `NPC energy`

## Queue Architecture

```
Producer (API) → Celery → Redis Broker
                ↓
    ┌───────────┼───────────┐
    crawl      process    analytics/export
    queue      queue       queue
```

Within crawlers: `CrawlQueue` (Redis ZSET) for URL-level work distribution.

## Scaling Strategies

| Scale | Crawlers | Workers | Notes |
|-------|----------|---------|-------|
| Dev | 1 | 1 Celery | Local Redis/PG |
| 10M posts/mo | 3 pods | 6 crawl workers | Separate browser pool per pod |
| 100M+ | K8s HPA | Sharded Redis | Proxy rotation, per-platform queues |

## Security & Compliance

- **Public data only** — no private groups/DMs
- **API-first** when platform provides official APIs
- **Rate limiting** — distributed Redis token bucket
- **No credential storage in code** — env vars / secrets manager
- **Robots/ToS** — operators must verify compliance per platform
- **PII** — strip @mentions; optional author_id hashing at storage layer

## Anti-Ban Strategies

1. Per-platform RPM limits in `platforms.yaml`
2. Exponential backoff + jitter on failures
3. Rotating User-Agent + optional proxy (`PROXY_URL`)
4. Playwright realistic viewport/locale (`vi-VN`)
5. `worker_prefetch_multiplier=1` — avoid burst patterns
6. Staggered Celery Beat schedules

## Monitoring

- **structlog** JSON logs in production
- **Flower** — Celery task monitoring (`:5555`)
- **Prometheus/Grafana** — extend existing `monitoring/` configs
- Key metrics: `items_collected`, `duplicate_rate`, `slang_candidates_found`

## Dataset Output Format

```json
{
  "text": "đỉnh nóc kịch trần 😭",
  "language": "vi",
  "source": "tiktok",
  "timestamp": "2026-05-20T10:00:00Z",
  "emotion": null,
  "toxicity": null,
  "sarcasm": null
}
```

## Hugging Face / LoRA Integration

```python
from data_engine.dataset.huggingface_integration import prepare_lora_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
ds = prepare_lora_dataset("./data/datasets/emotion_lens_social_v1", tokenizer)
# → use with PEFT LoRA + Trainer (see ai_nlp/training/emotion_pipeline/)
```

## Quick Start

```bash
# 1. Apply DB schema
psql -U emotionlens -d emotionlens -f database/init.sql
psql -U emotionlens -d emotionlens -f data_engine/database/init_data_engine.sql

# 2. Start stack
docker compose -f docker-compose.yml -f data_engine/docker-compose.data-engine.yml up -d

# 3. Trigger crawl
curl -X POST http://localhost:8100/api/v1/crawl \
  -H "Content-Type: application/json" \
  -d '{"platform": "reddit", "target": "vietnam", "params": {"limit": 100}}'
```
