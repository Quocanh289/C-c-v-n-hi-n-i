// ====================================================
// Popup UI - Quick Stats & Controls
// Minimal interface for on-the-fly monitoring
// ====================================================

import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { EmotionCategory, ExtensionStats, ExtensionSettings, DEFAULT_EMOTION_VISUALS } from '../types/emotion';

const Popup: React.FC = () => {
  const [stats, setStats] = useState<ExtensionStats | null>(null);
  const [settings, setSettings] = useState<ExtensionSettings | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [statsResult, settingsResult] = await Promise.all([
        chrome.runtime.sendMessage({ type: 'GET_STATS' }),
        chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }),
      ]);
      
      if (statsResult?.payload) setStats(statsResult.payload);
      if (settingsResult?.payload) setSettings(settingsResult.payload);
    } catch (error) {
      console.error('[EmotionLens] Failed to load popup data:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleExtension = async () => {
    if (!settings) return;
    const newEnabled = !settings.enabled;
    await chrome.runtime.sendMessage({
      type: 'UPDATE_SETTINGS',
      payload: { enabled: newEnabled },
    });
    setSettings({ ...settings, enabled: newEnabled });
  };

  const openOptions = () => {
    chrome.runtime.openOptionsPage?.() || 
      chrome.tabs.create({ url: 'dist/options/index.html' });
  };

  const openSidePanel = () => {
    chrome.sidePanel?.open?.()?.catch(() => {
      chrome.tabs.create({ url: 'dist/sidepanel/index.html' });
    });
  };

  if (loading) {
    return (
      <div className="popup-container loading">
        <div className="spinner" />
        <p>Loading Emotion Lens...</p>
      </div>
    );
  }

  return (
    <div className="popup-container">
      {/* Header */}
      <div className="popup-header">
        <div className="header-left">
          <span className="logo">🔍</span>
          <h1>Emotion Lens</h1>
        </div>
        <div className="header-right">
          <button
            className={`toggle-btn ${settings?.enabled ? 'active' : ''}`}
            onClick={toggleExtension}
            title={settings?.enabled ? 'Disable' : 'Enable'}
          >
            <div className="toggle-track">
              <div className="toggle-thumb" />
            </div>
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="stats-section">
        <div className="stat-card">
          <span className="stat-value">{stats?.totalAnalyzed || 0}</span>
          <span className="stat-label">Analyzed</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">
            {stats ? `${(stats.avgConfidence * 100).toFixed(0)}%` : '-'}
          </span>
          <span className="stat-label">Avg Confidence</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">
            {stats ? `${stats.avgInferenceTime.toFixed(0)}ms` : '-'}
          </span>
          <span className="stat-label">Avg Speed</span>
        </div>
      </div>

      {/* Emotion Breakdown */}
      <div className="emotions-section">
        <h2>Emotions Detected</h2>
        <div className="emotion-list">
          {Object.entries(DEFAULT_EMOTION_VISUALS)
            .filter(([key]) => key !== EmotionCategory.Neutral)
            .map(([key, visual]) => {
              const count = stats?.emotionsDetected?.[key as EmotionCategory] || 0;
              if (count === 0) return null;
              return (
                <div key={key} className="emotion-item">
                  <span className="emotion-icon">{visual.icon}</span>
                  <span className="emotion-label">{visual.label}</span>
                  <div className="emotion-bar-track">
                    <div
                      className="emotion-bar-fill"
                      style={{
                        width: `${(count / (stats?.totalAnalyzed || 1)) * 100}%`,
                        backgroundColor: visual.color,
                      }}
                    />
                  </div>
                  <span className="emotion-count">{count}</span>
                </div>
              );
            })}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="actions-section">
        <button className="action-btn" onClick={openOptions}>
          ⚙️ Settings
        </button>
        <button className="action-btn" onClick={openSidePanel}>
          📊 Details
        </button>
        <button
          className="action-btn danger"
          onClick={async () => {
            await chrome.runtime.sendMessage({ type: 'RESET_CACHE' });
            loadData();
          }}
        >
          🗑️ Reset
        </button>
      </div>

      {/* Footer */}
      <div className="popup-footer">
        <span className="footer-text">
          {stats?.localInferences || 0} local · {stats?.backendInferences || 0} cloud
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