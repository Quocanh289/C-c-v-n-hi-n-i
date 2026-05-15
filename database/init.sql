-- ====================================================
-- Emotion Lens Database Schema
-- PostgreSQL initialization script
-- ====================================================

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity search

-- ====================================================
-- Core Tables
-- ====================================================

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_hash VARCHAR(64) NOT NULL,
    text TEXT NOT NULL,
    primary_emotion VARCHAR(32) NOT NULL,
    emotion_scores JSONB NOT NULL DEFAULT '{}',
    toxicity_score FLOAT NOT NULL DEFAULT 0.0,
    toxicity_binary BOOLEAN NOT NULL DEFAULT FALSE,
    sarcasm_score FLOAT NOT NULL DEFAULT 0.0,
    sarcasm_binary BOOLEAN NOT NULL DEFAULT FALSE,
    intent VARCHAR(32),
    confidence FLOAT NOT NULL DEFAULT 0.0,
    language VARCHAR(8) NOT NULL DEFAULT 'en',
    source VARCHAR(16) NOT NULL DEFAULT 'backend',
    processing_time_ms FLOAT NOT NULL DEFAULT 0.0,
    platform VARCHAR(32),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Index for deduplication
    CONSTRAINT unique_text_hash UNIQUE (text_hash)
);

-- Indexes for analysis_results
CREATE INDEX idx_analysis_created_at ON analysis_results(created_at DESC);
CREATE INDEX idx_analysis_emotion ON analysis_results(primary_emotion);
CREATE INDEX idx_analysis_language ON analysis_results(language);
CREATE INDEX idx_analysis_platform ON analysis_results(platform);
CREATE INDEX idx_analysis_text_hash ON analysis_results(text_hash);

-- Feedback table for continuous learning
CREATE TABLE IF NOT EXISTS feedback_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    predicted_emotion VARCHAR(32) NOT NULL,
    corrected_emotion VARCHAR(32) NOT NULL,
    predicted_toxicity FLOAT,
    corrected_toxicity FLOAT,
    predicted_sarcasm FLOAT,
    corrected_sarcasm FLOAT,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    language VARCHAR(8) NOT NULL DEFAULT 'en',
    source VARCHAR(32) NOT NULL DEFAULT 'extension',
    is_used_in_training BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for feedback
CREATE INDEX idx_feedback_created_at ON feedback_data(created_at DESC);
CREATE INDEX idx_feedback_language ON feedback_data(language);
CREATE INDEX idx_feedback_trained ON feedback_data(is_used_in_training);

-- Slang terms dictionary
CREATE TABLE IF NOT EXISTS slang_terms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    term VARCHAR(128) NOT NULL UNIQUE,
    language VARCHAR(8) NOT NULL DEFAULT 'mixed',
    possible_emotions TEXT[] NOT NULL DEFAULT '{}',
    verified_emotion VARCHAR(32),
    frequency INTEGER NOT NULL DEFAULT 0,
    first_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    is_emerging BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for slang
CREATE INDEX idx_slang_term ON slang_terms(term);
CREATE INDEX idx_slang_verified ON slang_terms(is_verified);
CREATE INDEX idx_slang_emerging ON slang_terms(is_emerging);
CREATE INDEX idx_slang_frequency ON slang_terms(frequency DESC);

-- Unknown terms (potential emerging slang)
CREATE TABLE IF NOT EXISTS unknown_terms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    term VARCHAR(128) NOT NULL,
    context TEXT,
    frequency INTEGER NOT NULL DEFAULT 1,
    first_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for unknown terms
CREATE INDEX idx_unknown_term ON unknown_terms(term);
CREATE INDEX idx_unknown_frequency ON unknown_terms(frequency DESC);
CREATE INDEX idx_unknown_last_seen ON unknown_terms(last_seen DESC);

-- Model versions tracking
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(64) NOT NULL UNIQUE,
    model_path TEXT NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(16) NOT NULL DEFAULT 'staging',
    train_samples INTEGER DEFAULT 0,
    val_samples INTEGER DEFAULT 0,
    val_loss FLOAT,
    val_accuracy FLOAT,
    parent_version VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for model versions
CREATE INDEX idx_model_status ON model_versions(status);
CREATE INDEX idx_model_created ON model_versions(created_at DESC);

-- User settings (for syncing across browsers)
CREATE TABLE IF NOT EXISTS user_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(128) NOT NULL UNIQUE,
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ====================================================
-- Materialized Views for Analytics
-- ====================================================

-- Daily emotion summary
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_emotion_summary AS
SELECT
    DATE(created_at) AS analysis_date,
    primary_emotion,
    COUNT(*) AS count,
    AVG(confidence) AS avg_confidence,
    AVG(processing_time_ms) AS avg_processing_time
FROM analysis_results
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at), primary_emotion
ORDER BY analysis_date DESC, count DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_daily_emotion 
ON mv_daily_emotion_summary(analysis_date, primary_emotion);

-- Language distribution
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_language_distribution AS
SELECT
    language,
    COUNT(*) AS count,
    AVG(confidence) AS avg_confidence
FROM analysis_results
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY language
ORDER BY count DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_language ON mv_language_distribution(language);

-- ====================================================
-- Refresh Function for Materialized Views
-- ====================================================

CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_emotion_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_language_distribution;
END;
$$ LANGUAGE plpgsql;

-- ====================================================
-- Triggers
-- ====================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_slang_updated_at
    BEFORE UPDATE ON slang_terms
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_settings_updated_at
    BEFORE UPDATE ON user_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();