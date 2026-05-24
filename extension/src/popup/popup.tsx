// ====================================================
// Popup UI - active mode selector and label reference
// ====================================================

import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  EmotionCategory,
  ExtensionStats,
  ExtensionSettings,
  DEFAULT_SETTINGS,
  DetectionMode,
  GOEMOTIONS_28_VISUALS,
  COARSE_EMOTIONS_VISUALS,
  MENTAL_HEALTH_VISUALS,
} from '../types/emotion';

type PopupTab = 'en' | 'vi' | 'mh';

export default function Popup() {
  // --- Tất cả State đưa về chung 1 nơi ---
  const [userId, setUserId] = useState<string | null>(null);
  const [tab, setTab] = useState<PopupTab>('en');
  const [stats, setStats] = useState<ExtensionStats | null>(null);
  const [settings, setSettings] = useState<ExtensionSettings>(DEFAULT_SETTINGS);
  const [loading, setLoading] = useState(true);

  // --- Các hàm hỗ trợ ---
  const modeToTab = (mode: DetectionMode): PopupTab => {
    if (mode === 'mental_health_en') return 'mh';
    if (mode === 'emotion_vi') return 'vi';
    return 'en';
  };

  const tabToMode = (tab: PopupTab): DetectionMode => {
    if (tab === 'mh') return 'mental_health_en';
    if (tab === 'vi') return 'emotion_vi';
    return 'emotion_en';
  };

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

  const loadSettings = async () => {
    try {
      const result = await chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }) as any;
      if (result?.payload) {
        setSettings(result.payload);
        setTab(modeToTab(result.payload.activeMode || DEFAULT_SETTINGS.activeMode));
      }
    } catch (e) {
      console.error('[Popup] Failed to load settings:', e);
    }
  };

  const selectMode = async (nextTab: PopupTab) => {
    const updated = { ...settings, activeMode: tabToMode(nextTab) };
    setTab(nextTab);
    setSettings(updated);
    await chrome.runtime.sendMessage({ type: 'UPDATE_SETTINGS', payload: updated });
  };

  const handleOpenDashboard = () => {
    if (userId) {
      const url = `http://localhost:3000/dashboard?uid=${userId}`;
      chrome.tabs.create({ url }); // Mở tab mới
    }
  };

  // --- Hợp nhất useEffect khởi tạo ---
  useEffect(() => {
    // 1. Đọc userId từ storage
    chrome.storage.local.get(['userId'], (result) => {
      if (result.userId) {
        setUserId(result.userId);
      }
    });

    // 2. Tải thông số cấu hình và thống kê
    loadStats();
    loadSettings();
  }, []);

  // --- Giao diện hiển thị ---
  return (
    <div className="popup-container">
      <div className="popup-header">
        <div className="header-left">
          <span className="logo">AI</span>
          <h1>Emotion Lens</h1>
        </div>
        <span className="version-badge">v2.1</span>
      </div>

      {/* Khu vực hiển thị nút xem Dashboard */}
      <div className="p-4 w-64 border-b border-gray-700">
        <h1 className="font-bold text-lg mb-2">My Emotion Lens</h1>
        <p className="text-xs text-gray-400 mb-3">Your ID: {userId || 'Loading...'}</p>
        <button
          onClick={handleOpenDashboard}
          disabled={!userId}
          className="w-full bg-blue-600 text-white text-sm px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
        >
          View Analytics Dashboard
        </button>
      </div>

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
            <span className="stat-value">{stats.backendInferences}</span>
            <span className="stat-label">Backend</span>
          </div>
        </div>
      )}

      <div className="tab-bar">
        <button className={`tab-btn ${tab === 'en' ? 'active' : ''}`} onClick={() => selectMode('en')}>
          EN Emotion
        </button>
        <button className={`tab-btn ${tab === 'vi' ? 'active' : ''}`} onClick={() => selectMode('vi')}>
          VI Emotion
        </button>
        <button className={`tab-btn ${tab === 'mh' ? 'active' : ''}`} onClick={() => selectMode('mh')}>
          EN Mental
        </button>
      </div>

      <div className="model-info">
        <strong>{tab === 'mh' ? 'Mental health 7-label model' : 'Emotion detection model'}</strong>
        <br />
        {tab === 'mh' ? 'DeBERTa-v3 + LoRA through backend best_model' : 'One active mode scans posts and comments only'}
      </div>

      {tab === 'en' && (
        <div className="emotion-grid-28">
          {Object.entries(GOEMOTIONS_28_VISUALS).map(([key, visual]) => {
            const emotionMap: Record<string, EmotionCategory> = {
              admiration: EmotionCategory.Joy, amusement: EmotionCategory.Joy,
              anger: EmotionCategory.Anger, annoyance: EmotionCategory.Anger,
              approval: EmotionCategory.Joy, caring: EmotionCategory.Joy,
              confusion: EmotionCategory.Surprise, curiosity: EmotionCategory.Surprise,
              desire: EmotionCategory.Joy, disappointment: EmotionCategory.Sadness,
              disapproval: EmotionCategory.Anger, disgust: EmotionCategory.Anger,
              embarrassment: EmotionCategory.Sadness, excitement: EmotionCategory.Joy,
              fear: EmotionCategory.Fear, gratitude: EmotionCategory.Joy,
              grief: EmotionCategory.Sadness, joy: EmotionCategory.Joy,
              love: EmotionCategory.Joy, nervousness: EmotionCategory.Anxiety,
              optimism: EmotionCategory.Joy, pride: EmotionCategory.Joy,
              realization: EmotionCategory.Surprise, relief: EmotionCategory.Joy,
              remorse: EmotionCategory.Sadness, sadness: EmotionCategory.Sadness,
              surprise: EmotionCategory.Surprise, neutral: EmotionCategory.Neutral,
            };
            const detected = stats?.emotionsDetected[emotionMap[key] as EmotionCategory] || 0;
            return (
              <div key={key} className="emotion-card" style={{ borderLeftColor: visual.color, borderLeftWidth: 3 }}>
                <span className="emotion-icon">{visual.icon}</span>
                <div className="emotion-info">
                  <span className="emotion-name">{visual.label}</span>
                  <span className="emotion-group">{visual.group}</span>
                </div>
                {detected > 0 && <span className="emotion-count" style={{ color: visual.color }}>{detected}</span>}
              </div>
            );
          })}
        </div>
      )}

      {tab === 'vi' && (
        <div className="emotion-grid-9">
          {Object.entries(COARSE_EMOTIONS_VISUALS).map(([key, visual]) => (
            <div key={key} className="emotion-card coarse" style={{ borderLeftColor: visual.color, borderLeftWidth: 4 }}>
              <span className="emotion-icon">{visual.icon}</span>
              <span className="emotion-name">{visual.label}</span>
            </div>
          ))}
        </div>
      )}

      {tab === 'mh' && (
        <div className="emotion-grid-9">
          {Object.entries(MENTAL_HEALTH_VISUALS).map(([key, visual]) => (
            <div key={key} className="emotion-card coarse" style={{ borderLeftColor: visual.color, borderLeftWidth: 4 }}>
              <span className="emotion-icon">{visual.icon}</span>
              <div className="emotion-info">
                <span className="emotion-name">{visual.label}</span>
                <span className="emotion-group">{visual.severityLabel}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="actions-section">
        <button className="action-btn" onClick={() => chrome.runtime.openOptionsPage?.()}>
          Settings
        </button>
        <button className="action-btn" onClick={loadStats}>
          Refresh
        </button>
      </div>

      <div className="popup-footer">
        <span className="footer-text">Active mode: {tab === 'mh' ? 'Mental Health EN' : tab === 'vi' ? 'Emotion VI' : 'Emotion EN'}</span>
      </div>
    </div>
  );
}

// --- Phần render ra DOM được đưa ra ngoài cùng của file (Global Scope) ---
const rootElement = document.getElementById('root') || document.createElement('div');
if (!rootElement.id) {
  rootElement.id = 'root';
  document.body.appendChild(rootElement);
}
createRoot(rootElement).render(<Popup />);