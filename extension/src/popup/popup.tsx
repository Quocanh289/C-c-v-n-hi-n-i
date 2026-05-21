// ====================================================
// Popup UI - 28 Emotions for English, 9 for Vietnamese
// Shows detection stats + full emotion reference
// ====================================================

import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  EmotionCategory,
  ExtensionStats,
  ExtensionSettings,
  DEFAULT_EMOTION_VISUALS,
  GOEMOTIONS_28_VISUALS,
  COARSE_EMOTIONS_VISUALS,
} from '../types/emotion';

const Popup: React.FC = () => {
  const [tab, setTab] = useState<'en' | 'vi'>('en');
  const [stats, setStats] = useState<ExtensionStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const result = await chrome.runtime.sendMessage({ type: 'GET_STATS' });
      if ((result as any)?.payload) setStats((result as any).payload);
    } catch (e) {
      console.error('[Popup] Failed to load stats:', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="popup-container">
      {/* Header */}
      <div className="popup-header">
        <div className="header-left">
          <span className="logo">🔍</span>
          <h1>Emotion Lens</h1>
        </div>
        <span className="version-badge">v2.0</span>
      </div>

      {/* Stats Summary */}
      {!loading && stats && stats.totalAnalyzed > 0 && (
        <div className="stats-bar">
          <div className="stat-item">
            <span className="stat-value">{stats.totalAnalyzed}</span>
            <span className="stat-label">Analyzed</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{(stats.avgConfidence * 100).toFixed(0)}%</span>
            <span className="stat-label">Confidence</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{stats.avgInferenceTime.toFixed(0)}ms</span>
            <span className="stat-label">Speed</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{stats.localInferences}</span>
            <span className="stat-label">Local</span>
          </div>
        </div>
      )}

      {/* Tab Switcher */}
      <div className="tab-bar">
        <button
          className={`tab-btn ${tab === 'en' ? 'active' : ''}`}
          onClick={() => setTab('en')}
        >
          🇬🇧 EN · 28 Emotions
        </button>
        <button
          className={`tab-btn ${tab === 'vi' ? 'active' : ''}`}
          onClick={() => setTab('vi')}
        >
          🇻🇳 VI · 9 Emotions
        </button>
      </div>

      {/* Model Info */}
      <div className="model-info">
        <strong>GoEmotions 28-label model</strong>
        <br />
        XLM-RoBERTa + LoRA · Auto language detection
      </div>

      {/* English: 28 Emotions with detection counts */}
      {tab === 'en' && (
        <div className="emotion-grid-28">
          {Object.entries(GOEMOTIONS_28_VISUALS).map(([key, visual]) => {
            // Map 28-label to extension emotion key for stats lookup
            const emotionMap: Record<string, EmotionCategory> = {
              admiration: EmotionCategory.Joy,
              amusement: EmotionCategory.Joy,
              anger: EmotionCategory.Anger,
              annoyance: EmotionCategory.Anger,
              approval: EmotionCategory.Joy,
              caring: EmotionCategory.Joy,
              confusion: EmotionCategory.Surprise,
              curiosity: EmotionCategory.Surprise,
              desire: EmotionCategory.Joy,
              disappointment: EmotionCategory.Sadness,
              disapproval: EmotionCategory.Anger,
              disgust: EmotionCategory.Anger,
              embarrassment: EmotionCategory.Sadness,
              excitement: EmotionCategory.Joy,
              fear: EmotionCategory.Fear,
              gratitude: EmotionCategory.Joy,
              grief: EmotionCategory.Sadness,
              joy: EmotionCategory.Joy,
              love: EmotionCategory.Joy,
              nervousness: EmotionCategory.Anxiety,
              optimism: EmotionCategory.Joy,
              pride: EmotionCategory.Joy,
              realization: EmotionCategory.Surprise,
              relief: EmotionCategory.Joy,
              remorse: EmotionCategory.Sadness,
              sadness: EmotionCategory.Sadness,
              surprise: EmotionCategory.Surprise,
              neutral: EmotionCategory.Neutral,
            };
            const detected = stats?.emotionsDetected[emotionMap[key] as EmotionCategory] || 0;
            return (
              <div key={key} className="emotion-card" style={{ borderLeftColor: visual.color, borderLeftWidth: 3 }}>
                <span className="emotion-icon">{visual.icon}</span>
                <div className="emotion-info">
                  <span className="emotion-name">{visual.label}</span>
                  <span className="emotion-group">{visual.group}</span>
                </div>
                {detected > 0 && (
                  <span className="emotion-count" style={{ color: visual.color }}>{detected}</span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Vietnamese: 9 Emotions */}
      {tab === 'vi' && (
        <div className="emotion-grid-9">
          <div className="vi-notice">
            ⚠️ Vietnamese emotion data coming soon.
            <br />
            English model is currently active for all text.
          </div>
          {Object.entries(COARSE_EMOTIONS_VISUALS).map(([key, visual]) => {
            const emotionKey = key as EmotionCategory;
            const detected = stats?.emotionsDetected[emotionKey] || 0;
            return (
              <div key={key} className="emotion-card coarse" style={{ borderLeftColor: visual.color, borderLeftWidth: 4 }}>
                <span className="emotion-icon">{visual.icon}</span>
                <span className="emotion-name">{visual.label}</span>
                {detected > 0 && (
                  <span className="emotion-count" style={{ color: visual.color, fontSize: 11, fontWeight: 700 }}>{detected}</span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Quick Actions */}
      <div className="actions-section">
        <button className="action-btn" onClick={() => {
          if (chrome.runtime.openOptionsPage) {
            chrome.runtime.openOptionsPage();
          } else {
            chrome.tabs.create({ url: 'dist/options/index.html' });
          }
        }}>
          ⚙️ Settings
        </button>
        <button className="action-btn" onClick={() => {
          if (chrome.sidePanel?.open) {
            chrome.sidePanel.open().catch(() => {
              chrome.tabs.create({ url: 'dist/sidepanel/index.html' });
            });
          } else {
            chrome.tabs.create({ url: 'dist/sidepanel/index.html' });
          }
        }}>
          📊 Details
        </button>
        <button className="action-btn" onClick={loadStats}>
          🔄 Refresh
        </button>
      </div>

      {/* Footer */}
      <div className="popup-footer">
        <span className="footer-text">
          {stats ? `${stats.totalAnalyzed} texts analyzed · ` : ''}
          GoEmotions · 28 EN / 9 VI
        </span>
      </div>
    </div>
  );
};

// Mount
const root = document.createElement('div');
root.id = 'root';
document.body.appendChild(root);
createRoot(root).render(<Popup />);