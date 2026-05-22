// ====================================================
// Options/Settings Page
// ====================================================

import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ExtensionSettings,
  DEFAULT_SETTINGS,
  DetectionMode,
  GOEMOTIONS_28_VISUALS,
  COARSE_EMOTIONS_VISUALS,
  MENTAL_HEALTH_VISUALS,
} from '../types/emotion';

type OptionsTab = 'en' | 'vi' | 'mh';

const modeToTab = (mode: DetectionMode): OptionsTab => {
  if (mode === 'mental_health_en') return 'mh';
  if (mode === 'emotion_vi') return 'vi';
  return 'en';
};

const Options: React.FC = () => {
  const [settings, setSettings] = useState<ExtensionSettings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);
  const [tab, setTab] = useState<OptionsTab>('en');

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const result = await chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }) as any;
      if (result?.payload) {
        setSettings(result.payload);
        setTab(modeToTab(result.payload.activeMode || DEFAULT_SETTINGS.activeMode));
      }
    } catch (e) {
      console.warn('[Options] Using default settings', e);
    }
  };

  const saveSettings = async (partial: Partial<ExtensionSettings>) => {
    const updated = { ...settings, ...partial };
    setSettings(updated);
    if (partial.activeMode) setTab(modeToTab(partial.activeMode));
    try {
      await chrome.runtime.sendMessage({ type: 'UPDATE_SETTINGS', payload: updated });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      console.error('[Options] Failed to save', e);
    }
  };

  const modeButton = (mode: DetectionMode, label: string) => (
    <button
      className={`tab-btn ${settings.activeMode === mode ? 'active' : ''}`}
      onClick={() => saveSettings({ activeMode: mode })}
    >
      {label}
    </button>
  );

  return (
    <div className="options-container">
      <header className="options-header">
        <h1>Emotion Lens Settings</h1>
        {saved && <span className="saved-badge">Saved</span>}
      </header>

      <div className="model-banner">
        <strong>Active scan mode:</strong> {settings.activeMode === 'mental_health_en' ? 'English mental health' : settings.activeMode === 'emotion_vi' ? 'Vietnamese emotion' : 'English emotion'}
        <br />
        English mental health uses the backend DeBERTa-v3 + LoRA best_model checkpoint.
      </div>

      <section className="options-section">
        <h2>Active Mode</h2>
        <div className="tab-bar">
          {modeButton('emotion_en', 'English Emotion')}
          {modeButton('emotion_vi', 'Vietnamese Emotion')}
          {modeButton('mental_health_en', 'English Mental Health')}
        </div>
      </section>

      <section className="options-section">
        <h2>General</h2>
        {[
          { key: 'enabled', label: 'Extension Enabled' },
          { key: 'highlightEnabled', label: 'Show Highlights' },
          { key: 'labelsEnabled', label: 'Show Labels' },
          { key: 'toxicityFilterEnabled', label: 'Toxicity Filter' },
        ].map(({ key, label }) => (
          <div key={key} className="setting-row">
            <label>{label}</label>
            <input
              type="checkbox"
              checked={(settings as any)[key]}
              onChange={e => saveSettings({ [key]: e.target.checked } as any)}
            />
          </div>
        ))}
        <div className="setting-row">
          <label>Backend API URL</label>
          <input
            type="text"
            value={settings.backendApiUrl}
            onChange={e => saveSettings({ backendApiUrl: e.target.value })}
          />
        </div>
      </section>

      <section className="options-section">
        <h2>Sensitivity</h2>
        <div className="setting-row">
          <label>Confidence Threshold: {(settings.confidenceThreshold * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0"
            max="100"
            value={settings.confidenceThreshold * 100}
            onChange={e => saveSettings({ confidenceThreshold: parseInt(e.target.value, 10) / 100 })}
          />
        </div>
      </section>

      <section className="options-section">
        <h2>Labels</h2>
        <div className="tab-bar">
          <button className={`tab-btn ${tab === 'en' ? 'active' : ''}`} onClick={() => setTab('en')}>English 28</button>
          <button className={`tab-btn ${tab === 'vi' ? 'active' : ''}`} onClick={() => setTab('vi')}>Vietnamese 9</button>
          <button className={`tab-btn ${tab === 'mh' ? 'active' : ''}`} onClick={() => setTab('mh')}>Mental Health 7</button>
        </div>

        {tab === 'en' && (
          <div className="emotion-grid-28">
            {Object.entries(GOEMOTIONS_28_VISUALS).map(([key, visual]) => (
              <div key={key} className="emotion-card enabled" style={{ borderColor: visual.color }}>
                <span className="emotion-icon">{visual.icon}</span>
                <div className="emotion-card-info">
                  <span className="emotion-name">{visual.label}</span>
                  <span className="emotion-group">{visual.group}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {tab === 'vi' && (
          <div className="emotion-grid-9">
            {Object.entries(COARSE_EMOTIONS_VISUALS).map(([key, visual]) => (
              <div key={key} className="emotion-card coarse enabled" style={{ borderColor: visual.color }}>
                <span className="emotion-icon">{visual.icon}</span>
                <span className="emotion-name">{visual.label}</span>
              </div>
            ))}
          </div>
        )}

        {tab === 'mh' && (
          <div className="emotion-grid-9">
            {Object.entries(MENTAL_HEALTH_VISUALS).map(([key, visual]) => (
              <div key={key} className="emotion-card coarse enabled" style={{ borderColor: visual.color }}>
                <span className="emotion-icon">{visual.icon}</span>
                <div className="emotion-card-info">
                  <span className="emotion-name">{visual.label}</span>
                  <span className="emotion-group">{visual.severityLabel}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="options-section">
        <h2>Reset</h2>
        <button className="reset-btn" onClick={() => saveSettings(DEFAULT_SETTINGS)}>Reset to Defaults</button>
      </section>
    </div>
  );
};

const root = document.createElement('div');
root.id = 'root';
document.body.appendChild(root);
createRoot(root).render(<Options />);
