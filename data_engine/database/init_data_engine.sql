-- ====================================================
-- Data Engine Schema Extension
-- Run after database/init.sql (Emotion Lens base schema)
-- ====================================================

-- Crawl job tracking
CREATE TABLE IF NOT EXISTS crawl_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    platform VARCHAR(32) NOT NULL,
    target TEXT NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    params JSONB NOT NULL DEFAULT '{}',
    items_collected INTEGER NOT NULL DEFAULT 0,
    items_skipped INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    priority INTEGER NOT NULL DEFAULT 5,
    worker_id VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_crawl_jobs_platform ON crawl_jobs(platform);
CREATE INDEX idx_crawl_jobs_status ON crawl_jobs(status);
CREATE INDEX idx_crawl_jobs_created ON crawl_jobs(created_at DESC);

-- Dataset records for NLP training
CREATE TABLE IF NOT EXISTS dataset_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_hash VARCHAR(64) NOT NULL UNIQUE,
    text TEXT NOT NULL,
    text_raw TEXT NOT NULL,
    language VARCHAR(8) NOT NULL DEFAULT 'mixed',
    source VARCHAR(32) NOT NULL,
    platform_post_id VARCHAR(128),
    url TEXT,
    emotion VARCHAR(32),
    toxicity FLOAT,
    sarcasm FLOAT,
    labels JSONB NOT NULL DEFAULT '{}',
    is_spam BOOLEAN NOT NULL DEFAULT FALSE,
    is_labeled BOOLEAN NOT NULL DEFAULT FALSE,
    dataset_version VARCHAR(16) NOT NULL DEFAULT 'v1',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dataset_language ON dataset_records(language);
CREATE INDEX idx_dataset_source ON dataset_records(source);
CREATE INDEX idx_dataset_labeled ON dataset_records(is_labeled);
CREATE INDEX idx_dataset_version ON dataset_records(dataset_version);
CREATE INDEX idx_dataset_created ON dataset_records(created_at DESC);
CREATE INDEX idx_dataset_text_trgm ON dataset_records USING gin (text gin_trgm_ops);

-- Slang drift candidates (human review workflow)
CREATE TABLE IF NOT EXISTS slang_candidates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    term VARCHAR(128) NOT NULL,
    language VARCHAR(8) NOT NULL DEFAULT 'mixed',
    frequency_current INTEGER NOT NULL DEFAULT 0,
    frequency_previous INTEGER NOT NULL DEFAULT 0,
    growth_rate FLOAT NOT NULL DEFAULT 0.0,
    contexts JSONB NOT NULL DEFAULT '[]',
    status VARCHAR(16) NOT NULL DEFAULT 'candidate',
    reviewed_by VARCHAR(128),
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_slang_candidates_term ON slang_candidates(term);
CREATE INDEX idx_slang_candidates_status ON slang_candidates(status);
CREATE INDEX idx_slang_candidates_growth ON slang_candidates(growth_rate DESC);

-- Dataset versioning
CREATE TABLE IF NOT EXISTS dataset_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(32) NOT NULL UNIQUE,
    record_count INTEGER NOT NULL DEFAULT 0,
    export_path TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Annotation tasks for labeling pipeline
CREATE TABLE IF NOT EXISTS annotation_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_record_id UUID REFERENCES dataset_records(id),
    priority_score FLOAT NOT NULL DEFAULT 0.5,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    assigned_to VARCHAR(128),
    label_studio_task_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_annotation_status ON annotation_tasks(status);
CREATE INDEX idx_annotation_priority ON annotation_tasks(priority_score DESC);

-- Multi-annotator consensus results
CREATE TABLE IF NOT EXISTS annotation_consensus (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_record_id UUID REFERENCES dataset_records(id),
    emotion VARCHAR(32),
    toxicity FLOAT,
    sarcasm FLOAT,
    agreement_score FLOAT NOT NULL DEFAULT 0.0,
    annotator_count INTEGER NOT NULL DEFAULT 0,
    votes JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
