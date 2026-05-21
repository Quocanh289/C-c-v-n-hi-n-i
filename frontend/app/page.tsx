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
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'en' | 'vi'>('en');
  const [error, setError] = useState<string | null>(null);

  const analyzeText = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    
    const detectedLang = detectLanguage(text);
    setActiveTab(detectedLang);
    
    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          return_all_probs: true,
          output_mode: detectedLang === 'vi' ? 'coarse' : 'fine',
        }),
      });
      
      if (!response.ok) throw new Error(`API error: ${response.status}`);
      
      const data = await response.json();
      
      // Build structured result
      const result: AnalysisResult = {
        primary_emotion: data.primary_emotion || 'neutral',
        confidence: data.confidence || 0,
        label_type: detectedLang === 'vi' ? 'coarse' : 'fine',
        language: data.language || detectedLang,
        scores_28: data.scores_28 || {},
        scores_9: data.scores_9 || {},
        source: data.source || 'backend',
        num_labels: detectedLang === 'vi' ? 9 : 28,
      };
      
      setResult(result);
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
  const showFine = activeTab === 'en';
  const scores = showFine ? result?.scores_28 : result?.scores_9;
  const labels = showFine ? GOEMOTIONS_28 : COARSE_EMOTIONS;

  return (
    <main style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)', color: '#e2e8f0', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '40px 20px' }}>
        {/* Header */}
        <header style={{ textAlign: 'center', marginBottom: 40 }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 700, background: 'linear-gradient(135deg, #22c55e, #3b82f6, #a855f7)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: 8 }}>
            Emotion Lens Analyzer
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '1.05rem' }}>
            28 emotions for English · 9 emotions for Vietnamese
          </p>
        </header>

        {/* Input Area */}
        <div style={{ background: '#1e293b', borderRadius: 16, padding: 24, border: '1px solid #334155', marginBottom: 24 }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', marginBottom: 12 }}>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: 6, fontWeight: 500 }}>
                Enter text to analyze emotion
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type something in English or Vietnamese..."
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
          
          {/* Language indicator */}
          {text.trim() && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>Detected:</span>
              <span style={{ 
                padding: '3px 10px', 
                borderRadius: 6, 
                fontSize: '0.78rem', 
                fontWeight: 600,
                background: detectedLang === 'vi' ? 'rgba(249, 115, 22, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                color: detectedLang === 'vi' ? '#fb923c' : '#60a5fa',
              }}>
                {detectedLang === 'vi' ? '🇻🇳 Vietnamese (9 emotions)' : '🇬🇧 English (28 emotions)'}
              </span>
              <span style={{ fontSize: '0.75rem', color: '#475569' }}>
                {detectedLang === 'en' ? '28 fine-grained GoEmotions labels' : '9 coarse labels (Vietnamese data coming soon)'}
              </span>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 12, padding: 16, marginBottom: 24, color: '#fca5a5' }}>
            ⚠️ {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <>
            {/* Primary Emotion */}
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

            {/* Tab Switcher */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
              <button
                onClick={() => setActiveTab('en')}
                style={{
                  padding: '10px 20px',
                  borderRadius: 8,
                  border: '1px solid',
                  borderColor: activeTab === 'en' ? '#3b82f6' : '#334155',
                  background: activeTab === 'en' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                  color: activeTab === 'en' ? '#60a5fa' : '#94a3b8',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                🇬🇧 English · 28 Emotions
              </button>
              <button
                onClick={() => setActiveTab('vi')}
                style={{
                  padding: '10px 20px',
                  borderRadius: 8,
                  border: '1px solid',
                  borderColor: activeTab === 'vi' ? '#f97316' : '#334155',
                  background: activeTab === 'vi' ? 'rgba(249, 115, 22, 0.15)' : 'transparent',
                  color: activeTab === 'vi' ? '#fb923c' : '#94a3b8',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                🇻🇳 Vietnamese · 9 Emotions
              </button>
            </div>

            {/* Emotion Bars - 28 Labels (English) */}
            {activeTab === 'en' && scores && (
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
            {activeTab === 'vi' && scores && (
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
                    
                    // Show which 28 sub-emotions map to this coarse emotion
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
        {!result && !loading && !error && (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: '#475569' }}>
            <div style={{ fontSize: '4rem', marginBottom: 16 }}>🔍</div>
            <p style={{ fontSize: '1.1rem', marginBottom: 8 }}>Enter text above and click Analyze</p>
            <p style={{ fontSize: '0.9rem' }}>Press Ctrl+Enter to analyze quickly</p>
          </div>
        )}

        {/* Footer */}
        <footer style={{ textAlign: 'center', marginTop: 60, padding: '20px 0', borderTop: '1px solid #1e293b', color: '#475569', fontSize: '0.85rem' }}>
          <p>Emotion Lens · GoEmotions 28-label model (XLM-RoBERTa + LoRA)</p>
          <p style={{ marginTop: 4 }}>English: 28 fine-grained emotions · Vietnamese: 9 coarse emotions (no training data yet)</p>
        </footer>
      </div>
    </main>
  );
}