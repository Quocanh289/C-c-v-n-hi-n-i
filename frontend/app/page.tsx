"use client";

import React, { useState, useCallback } from 'react';

// ============================================================
// Label Definitions
// ============================================================

const GOEMOTIONS_28 = [
  "admiration", "amusement", "anger", "annoyance", "approval",
  "caring", "confusion", "curiosity", "desire", "disappointment",
  "disapproval", "disgust", "embarrassment", "excitement", "fear",
  "gratitude", "grief", "joy", "love", "nervousness",
  "optimism", "pride", "realization", "relief", "remorse",
  "sadness", "surprise", "neutral",
];

// Mental Health Conditions
const MENTAL_HEALTH_LABELS = [
  "Normal",
  "Depression",
  "Anxiety",
  "Bipolar",
  "Stress",
  "Suicidal",
  "Personality_disorder",
];

const MH_COLORS: Record<string, string> = {
  "Normal": "#22c55e",
  "Depression": "#3b82f6",
  "Anxiety": "#f97316",
  "Bipolar": "#a855f7",
  "Stress": "#ef4444",
  "Suicidal": "#dc2626",
  "Personality_disorder": "#ec4899",
};

const MH_ICONS: Record<string, string> = {
  "Normal": "😊",
  "Depression": "😢",
  "Anxiety": "😰",
  "Bipolar": "🔮",
  "Stress": "😫",
  "Suicidal": "💔",
  "Personality_disorder": "🧩",
};

const MH_SEVERITY_MAP: Record<string, number> = {
  "Normal": 0,
  "Stress": 1,
  "Anxiety": 2,
  "Personality_disorder": 2,
  "Bipolar": 3,
  "Depression": 4,
  "Suicidal": 5,
};

const MH_SEVERITY_LABELS: Record<number, string> = {
  0: "Healthy",
  1: "Mild",
  2: "Moderate",
  3: "Moderate-High",
  4: "Severe",
  5: "Critical",
};

const SEVERITY_COLORS: Record<number, string> = {
  0: "#22c55e",
  1: "#84cc16",
  2: "#f97316",
  3: "#ef4444",
  4: "#dc2626",
  5: "#7f1d1d",
};

const COARSE_EMOTIONS = [
  "admiration", "anger", "anxiety", "fear",
  "joy", "love", "sadness", "surprise", "neutral",
];

const EMOTION_28_TO_9_MAP: Record<string, string> = {
  "admiration": "admiration", "amusement": "joy", "anger": "anger",
  "annoyance": "anger", "approval": "admiration", "caring": "love",
  "confusion": "surprise", "curiosity": "surprise", "desire": "admiration",
  "disappointment": "sadness", "disapproval": "anger", "disgust": "anger",
  "embarrassment": "sadness", "excitement": "joy", "fear": "fear",
  "gratitude": "admiration", "grief": "sadness", "joy": "joy",
  "love": "love", "nervousness": "anxiety", "optimism": "admiration",
  "pride": "joy", "realization": "surprise", "relief": "joy",
  "remorse": "sadness", "sadness": "sadness", "surprise": "surprise",
  "neutral": "neutral",
};

const GROUP_COLORS: Record<string, string> = {
  admiration: '#f59e0b',
  anger:      '#ef4444',
  anxiety:    '#f97316',
  fear:       '#7c3aed',
  joy:        '#22c55e',
  love:       '#ec4899',
  sadness:    '#3b82f6',
  surprise:   '#a855f7',
  neutral:    '#6b7280',
};

const GROUP_ICONS: Record<string, string> = {
  admiration: '👏',
  anger:      '😡',
  anxiety:    '😰',
  fear:       '😨',
  joy:        '😊',
  love:       '❤️',
  sadness:    '😢',
  surprise:   '😲',
  neutral:    '😐',
};

const EN_EMOTION_ICONS: Record<string, string> = {
  admiration: '👏', amusement: '😂', anger: '😡', annoyance: '😤',
  approval: '👍', caring: '💚', confusion: '😕', curiosity: '🤔',
  desire: '😍', disappointment: '😞', disapproval: '👎', disgust: '🤢',
  embarrassment: '😳', excitement: '🤩', fear: '😨', gratitude: '🙏',
  grief: '😭', joy: '😊', love: '❤️', nervousness: '😬',
  optimism: '🌟', pride: '🦁', realization: '💡', relief: '😌',
  remorse: '😔', sadness: '😢', surprise: '😲', neutral: '😐',
};

// Mental health result type
type MentalHealthResult = {
  primary_condition: string;
  primary_confidence: number;
  all_scores: Record<string, number>;
  needs_attention: boolean;
  severity_level: number;
  severity_label: string;
  top_predictions: { label: string; confidence: number }[];
  source: string;
  num_labels: number;
};

type AnalysisResult = {
  primary_emotion: string;
  confidence: number;
  label_type: string;
  language: string;
  scores_28?: Record<string, number>;
  scores_9?: Record<string, number>;
  source: string;
  num_labels: number;
};

const VIETNAMESE_CHARS_REGEX = /[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]/i;

function detectLanguage(text: string): 'vi' | 'en' {
  const viCount = (text.match(VIETNAMESE_CHARS_REGEX) || []).length;
  const totalChars = text.replace(/\s/g, '').length;
  if (totalChars === 0) return 'en';
  return (viCount / totalChars) > 0.15 ? 'vi' : 'en';
}

export default function Home() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [mhResult, setMhResult] = useState<MentalHealthResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'emotion_en' | 'emotion_vi' | 'mental_health'>('emotion_en');
  const [error, setError] = useState<string | null>(null);

  const analyzeText = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setMhResult(null);
    
    const detectedLang = detectLanguage(text);
    const isVietnamese = detectedLang === 'vi';
    setActiveTab(isVietnamese ? 'emotion_vi' : 'emotion_en');
    
    try {
      // Parallel calls: emotion analysis + mental health analysis
      const [emotionRes, mhRes] = await Promise.all([
        fetch('http://localhost:8000/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: text,
            return_all_probs: true,
            output_mode: isVietnamese ? 'coarse' : 'fine',
          }),
        }),
        fetch('http://localhost:8000/api/mental-health/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text }),
        }),
      ]);
      
      if (!emotionRes.ok) throw new Error(`Emotion API error: ${emotionRes.status}`);
      
      const emotionData = await emotionRes.json();
      let mhData: MentalHealthResult | null = null;
      if (mhRes.ok) {
        mhData = await mhRes.json();
      }
      
      const emotionResult: AnalysisResult = {
        primary_emotion: emotionData.primary_emotion || 'neutral',
        confidence: emotionData.confidence || 0,
        label_type: isVietnamese ? 'coarse' : 'fine',
        language: emotionData.language || detectedLang,
        scores_28: emotionData.scores_28 || {},
        scores_9: emotionData.scores_9 || {},
        source: emotionData.source || 'backend',
        num_labels: isVietnamese ? 9 : 28,
      };
      
      setResult(emotionResult);
      if (mhData) setMhResult(mhData);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  }, [text]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      analyzeText();
    }
  };

  const detectedLang = text.trim() ? detectLanguage(text) : 'en';
  const isMentalHealthTab = activeTab === 'mental_health';
  const showFine = activeTab === 'emotion_en';
  const isEmotionVi = activeTab === 'emotion_vi';
  const scores = showFine ? result?.scores_28 : result?.scores_9;
  const labels = showFine ? GOEMOTIONS_28 : COARSE_EMOTIONS;

  const getSeverityBarColor = (sev: number) => {
    const colors = ['#22c55e', '#84cc16', '#f97316', '#ef4444', '#dc2626', '#7f1d1d'];
    return colors[Math.min(sev, 5)];
  };

  return (
    <main style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)', color: '#e2e8f0', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '40px 20px' }}>
        {/* Header */}
        <header style={{ textAlign: 'center', marginBottom: 32 }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 700, background: 'linear-gradient(135deg, #22c55e, #3b82f6, #a855f7)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: 8 }}>
            Emotion Lens Analyzer
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '1.05rem' }}>
            28 emotions · 7 mental health conditions
          </p>
        </header>

        {/* Input Area */}
        <div style={{ background: '#1e293b', borderRadius: 16, padding: 24, border: '1px solid #334155', marginBottom: 24 }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', marginBottom: 12 }}>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: 6, fontWeight: 500 }}>
                Enter text to analyze
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type something in English..."
                rows={4}
                style={{
                  width: '100%',
                  padding: '14px 16px',
                  borderRadius: 10,
                  border: '1px solid #334155',
                  background: '#0f172a',
                  color: '#e2e8f0',
                  fontSize: '1rem',
                  resize: 'vertical',
                  outline: 'none',
                }}
              />
            </div>
            <button
              onClick={analyzeText}
              disabled={loading || !text.trim()}
              style={{
                padding: '14px 32px',
                borderRadius: 10,
                border: 'none',
                background: loading ? '#334155' : 'linear-gradient(135deg, #22c55e, #16a34a)',
                color: 'white',
                fontSize: '1rem',
                fontWeight: 600,
                cursor: loading || !text.trim() ? 'not-allowed' : 'pointer',
                whiteSpace: 'nowrap',
                opacity: loading || !text.trim() ? 0.6 : 1,
                transition: 'all 0.2s',
              }}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 12, padding: 16, marginBottom: 24, color: '#fca5a5' }}>
            ⚠️ {error}
          </div>
        )}

        {/* Tab Switcher */}
        {(result || mhResult) && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            <button
              onClick={() => setActiveTab('emotion_en')}
              style={{
                padding: '10px 20px',
                borderRadius: 8,
                border: '1px solid',
                borderColor: activeTab === 'emotion_en' ? '#3b82f6' : '#334155',
                background: activeTab === 'emotion_en' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                color: activeTab === 'emotion_en' ? '#60a5fa' : '#94a3b8',
                fontSize: '0.85rem',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              🇬🇧 28 Emotions (EN)
            </button>
            <button
              onClick={() => setActiveTab('emotion_vi')}
              style={{
                padding: '10px 20px',
                borderRadius: 8,
                border: '1px solid',
                borderColor: activeTab === 'emotion_vi' ? '#f97316' : '#334155',
                background: activeTab === 'emotion_vi' ? 'rgba(249, 115, 22, 0.15)' : 'transparent',
                color: activeTab === 'emotion_vi' ? '#fb923c' : '#94a3b8',
                fontSize: '0.85rem',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              🇻🇳 9 Emotions (VI)
            </button>
            <button
              onClick={() => setActiveTab('mental_health')}
              style={{
                padding: '10px 20px',
                borderRadius: 8,
                border: '1px solid',
                borderColor: activeTab === 'mental_health' ? '#a855f7' : '#334155',
                background: activeTab === 'mental_health' ? 'rgba(168, 85, 247, 0.15)' : 'transparent',
                color: activeTab === 'mental_health' ? '#c084fc' : '#94a3b8',
                fontSize: '0.85rem',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              🧠 7 Mental Health Conditions
            </button>
          </div>
        )}

        {/* Mental Health Results */}
        {isMentalHealthTab && mhResult && (
          <>
            {/* Severity Banner */}
            <div style={{
              background: `linear-gradient(135deg, ${getSeverityBarColor(mhResult.severity_level)}22, #1e293b)`,
              borderRadius: 16,
              padding: '24px 32px',
              border: `1px solid ${getSeverityBarColor(mhResult.severity_level)}44`,
              marginBottom: 24,
              display: 'flex',
              alignItems: 'center',
              gap: 20,
            }}>
              <div style={{ fontSize: '3rem', lineHeight: 1 }}>
                {MH_ICONS[mhResult.primary_condition] || '😐'}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: 4 }}>PRIMARY MENTAL HEALTH CONDITION</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700 }}>
                  {mhResult.primary_condition.replace(/_/g, ' ')}
                </div>
                <div style={{ marginTop: 8, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                    Confidence: <span style={{ color: MH_COLORS[mhResult.primary_condition] || '#22c55e', fontWeight: 600 }}>{(mhResult.primary_confidence * 100).toFixed(1)}%</span>
                  </span>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                    Severity: <span style={{ color: SEVERITY_COLORS[mhResult.severity_level] || '#6b7280', fontWeight: 600 }}>
                      {MH_SEVERITY_LABELS[mhResult.severity_level] || 'Unknown'}
                    </span>
                  </span>
                  {mhResult.needs_attention && (
                    <span style={{ padding: '2px 10px', borderRadius: 6, fontSize: '0.8rem', fontWeight: 600, background: 'rgba(220, 38, 38, 0.2)', color: '#fca5a5' }}>
                      ⚠️ Needs Attention
                    </span>
                  )}
                  {!mhResult.needs_attention && (
                    <span style={{ padding: '2px 10px', borderRadius: 6, fontSize: '0.8rem', fontWeight: 600, background: 'rgba(34, 197, 94, 0.15)', color: '#86efac' }}>
                      ✅ Healthy
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Severity Bar */}
            <div style={{ background: '#1e293b', borderRadius: 12, padding: 20, border: '1px solid #334155', marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Severity Level</span>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: SEVERITY_COLORS[mhResult.severity_level] }}>
                  {MH_SEVERITY_LABELS[mhResult.severity_level]}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 4, height: 24 }}>
                {[0, 1, 2, 3, 4, 5].map(level => (
                  <div
                    key={level}
                    style={{
                      flex: 1,
                      borderRadius: 6,
                      background: mhResult.severity_level >= level
                        ? SEVERITY_COLORS[level]
                        : '#1e293b',
                      border: `1px solid ${SEVERITY_COLORS[level]}44`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      color: mhResult.severity_level >= level ? 'white' : '#475569',
                      transition: 'all 0.3s',
                    }}
                  >
                    {MH_SEVERITY_LABELS[level].slice(0, 4)}
                  </div>
                ))}
              </div>
            </div>

            {/* All Mental Health Scores */}
            <div style={{ background: '#1e293b', borderRadius: 16, padding: 24, border: '1px solid #334155' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16, color: '#e2e8f0' }}>
                All Condition Scores
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: 10 }}>
                {MENTAL_HEALTH_LABELS.map((label) => {
                  const score = mhResult.all_scores[label] || 0;
                  const color = MH_COLORS[label] || '#6b7280';
                  const icon = MH_ICONS[label] || '😐';
                  const barPct = Math.min(100, Math.round(score * 100));
                  const isActive = label === mhResult.primary_condition;
                  const severity = MH_SEVERITY_MAP[label] || 0;
                  
                  return (
                    <div key={label} style={{
                      background: '#0f172a',
                      borderRadius: 12,
                      padding: '14px 16px',
                      border: `1px solid ${isActive ? color : '#1e293b'}`,
                      borderLeft: `4px solid ${color}`,
                      transition: 'all 0.2s',
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <span style={{ fontSize: '1.3rem' }}>{icon}</span>
                          <div>
                            <span style={{ fontSize: '1rem', fontWeight: 700, textTransform: 'capitalize' }}>
                              {label.replace(/_/g, ' ')}
                            </span>
                            <span style={{ fontSize: '0.7rem', color: '#64748b', marginLeft: 8 }}>
                              {MH_SEVERITY_LABELS[severity]}
                            </span>
                          </div>
                        </div>
                        <span style={{ fontSize: '1.1rem', fontWeight: 800, color }}>
                          {barPct}%
                        </span>
                      </div>
                      <div style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${barPct}%`,
                          background: color,
                          borderRadius: 4,
                          opacity: isActive ? 1 : 0.5,
                          transition: 'width 0.5s ease',
                        }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Top 3 Predictions */}
            <div style={{ background: '#1e293b', borderRadius: 16, padding: 20, border: '1px solid #334155', marginTop: 16 }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 12, color: '#e2e8f0' }}>Top Predictions</h3>
              <div style={{ display: 'flex', gap: 12 }}>
                {mhResult.top_predictions.map((pred, i) => (
                  <div key={pred.label} style={{
                    flex: 1,
                    background: '#0f172a',
                    borderRadius: 10,
                    padding: '12px 16px',
                    border: `1px solid ${i === 0 ? MH_COLORS[pred.label] : '#1e293b'}`,
                  }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: 4 }}>
                      {i === 0 ? '🏆 Primary' : i === 1 ? '🥈 Secondary' : '🥉 Third'}
                    </div>
                    <div style={{ fontSize: '1rem', fontWeight: 700, color: MH_COLORS[pred.label] }}>
                      {pred.label.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: 4 }}>
                      {(pred.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model info */}
            <div style={{ textAlign: 'center', marginTop: 16, padding: '12px', color: '#475569', fontSize: '0.8rem' }}>
              Mental health model: DeBERTa-v3-base + LoRA · 7 conditions · {mhResult.source}
            </div>
          </>
        )}

        {/* Emotion Results Header */}
        {!isMentalHealthTab && result && (
          <>
            {/* Primary Emotion Header */}
            <div style={{ 
              background: 'linear-gradient(135deg, #1e293b, #0f172a)', 
              borderRadius: 16, 
              padding: '24px 32px', 
              border: '1px solid #334155',
              marginBottom: 24,
              display: 'flex',
              alignItems: 'center',
              gap: 20,
            }}>
              <div style={{ fontSize: '3rem', lineHeight: 1 }}>
                {EN_EMOTION_ICONS[result.primary_emotion] || '😐'}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: 4 }}>PRIMARY EMOTION</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, textTransform: 'capitalize' }}>
                  {result.primary_emotion.replace(/_/g, ' ')}
                </div>
                <div style={{ marginTop: 8, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                    Confidence: <span style={{ color: '#22c55e', fontWeight: 600 }}>{(result.confidence * 100).toFixed(1)}%</span>
                  </span>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                    Mode: <span style={{ color: '#60a5fa', fontWeight: 600 }}>{result.label_type === 'fine' ? '28 Labels (EN)' : '9 Labels (VI)'}</span>
                  </span>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                    Language: <span style={{ color: '#a78bfa', fontWeight: 600 }}>{result.language.toUpperCase()}</span>
                  </span>
                </div>
              </div>
            </div>

            {/* Emotion Bars - 28 Labels (English) */}
            {showFine && scores && (
              <div style={{ background: '#1e293b', borderRadius: 16, padding: 24, border: '1px solid #334155' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16, color: '#e2e8f0' }}>
                  28 Fine-Grained Emotion Scores
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 8 }}>
                  {GOEMOTIONS_28.map((label) => {
                    const score = (scores as Record<string, number>)[label] || 0;
                    const group = EMOTION_28_TO_9_MAP[label];
                    const groupColor = GROUP_COLORS[group];
                    const icon = EN_EMOTION_ICONS[label] || '😐';
                    const barPct = Math.min(100, Math.round(score * 100));
                    
                    return (
                      <div key={label} style={{
                        background: '#0f172a',
                        borderRadius: 10,
                        padding: '10px 14px',
                        border: '1px solid #1e293b',
                        transition: 'all 0.2s',
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontSize: '1.1rem' }}>{icon}</span>
                            <span style={{ fontSize: '0.85rem', fontWeight: 600, textTransform: 'capitalize' }}>
                              {label}
                            </span>
                            <span style={{ fontSize: '0.7rem', color: '#64748b', padding: '1px 6px', borderRadius: 4, background: 'rgba(255,255,255,0.05)' }}>
                              {group}
                            </span>
                          </div>
                          <span style={{ fontSize: '0.85rem', fontWeight: 700, color: groupColor }}>
                            {barPct}%
                          </span>
                        </div>
                        <div style={{ height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{
                            height: '100%',
                            width: `${barPct}%`,
                            background: groupColor,
                            borderRadius: 3,
                            opacity: score > 0.3 ? 1 : 0.4,
                            transition: 'width 0.5s ease',
                          }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Emotion Bars - 9 Labels (Vietnamese) */}
            {isEmotionVi && scores && (
              <div style={{ background: '#1e293b', borderRadius: 16, padding: 24, border: '1px solid #334155' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#e2e8f0' }}>
                    9 Coarse Emotion Scores
                  </h3>
                  <span style={{ fontSize: '0.8rem', color: '#fb923c', background: 'rgba(249, 115, 22, 0.1)', padding: '4px 12px', borderRadius: 6 }}>
                    Aggregated from 28-label model
                  </span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: 10 }}>
                  {COARSE_EMOTIONS.map((label) => {
                    const score = (scores as Record<string, number>)[label] || 0;
                    const color = GROUP_COLORS[label];
                    const icon = GROUP_ICONS[label];
                    const barPct = Math.min(100, Math.round(score * 100));
                    const subEmotions = GOEMOTIONS_28.filter(e => EMOTION_28_TO_9_MAP[e] === label);
                    
                    return (
                      <div key={label} style={{
                        background: '#0f172a',
                        borderRadius: 12,
                        padding: '14px 16px',
                        border: '1px solid #1e293b',
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <span style={{ fontSize: '1.3rem' }}>{icon}</span>
                            <span style={{ fontSize: '1rem', fontWeight: 700, textTransform: 'capitalize' }}>
                              {label}
                            </span>
                          </div>
                          <span style={{ fontSize: '1rem', fontWeight: 700, color }}>
                            {barPct}%
                          </span>
                        </div>
                        <div style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden', marginBottom: 6 }}>
                          <div style={{
                            height: '100%',
                            width: `${barPct}%`,
                            background: color,
                            borderRadius: 4,
                            transition: 'width 0.5s ease',
                          }} />
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#475569' }}>
                          Includes: {subEmotions.join(', ')}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}

        {/* No result state */}
        {!result && !mhResult && !loading && !error && (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: '#475569' }}>
            <div style={{ fontSize: '4rem', marginBottom: 16 }}>🔍</div>
            <p style={{ fontSize: '1.1rem', marginBottom: 8 }}>Enter text above and click Analyze</p>
            <p style={{ fontSize: '0.9rem' }}>Press Ctrl+Enter to analyze quickly</p>
            <div style={{ marginTop: 24, display: 'flex', justifyContent: 'center', gap: 24, flexWrap: 'wrap' }}>
              <div style={{ background: '#1e293b', borderRadius: 12, padding: '16px 24px', border: '1px solid #334155', maxWidth: 300 }}>
                <div style={{ fontSize: '1.5rem', marginBottom: 8 }}>😊</div>
                <strong style={{ color: '#e2e8f0' }}>Emotion Detection</strong>
                <p style={{ fontSize: '0.85rem', marginTop: 4 }}>28 fine-grained emotions (English) · 9 coarse (Vietnamese) · GoEmotions model</p>
              </div>
              <div style={{ background: '#1e293b', borderRadius: 12, padding: '16px 24px', border: '1px solid #334155', maxWidth: 300 }}>
                <div style={{ fontSize: '1.5rem', marginBottom: 8 }}>🧠</div>
                <strong style={{ color: '#e2e8f0' }}>Mental Health Screening</strong>
                <p style={{ fontSize: '0.85rem', marginTop: 4 }}>7 conditions · DeBERTa-v3 + LoRA · severity assessment</p>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer style={{ textAlign: 'center', marginTop: 60, padding: '20px 0', borderTop: '1px solid #1e293b', color: '#475569', fontSize: '0.85rem' }}>
          <p>Emotion Lens · GoEmotions 28-label (XLM-RoBERTa + LoRA) · Mental Health 7-class (DeBERTa-v3 + LoRA)</p>
          <p style={{ marginTop: 4 }}>Run training: <code style={{ background: '#1e293b', padding: '2px 8px', borderRadius: 4 }}>python -m ai_nlp.training.mental_health_pipeline.run --mode train</code></p>
        </footer>
      </div>
    </main>
  );
}