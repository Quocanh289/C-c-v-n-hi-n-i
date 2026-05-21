// ====================================================
// Options/Settings Page - Full Emotion Lens Settings
// Shows 28 emotions for English, 9 for Vietnamese
// ====================================================

import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  EmotionCategory,
  ExtensionSettings,
  DEFAULT_SETTINGS,
  DEFAULT_EMOTION_VISUALS,
  GOEMOTIONS_28_VISUALS,
  COARSE_EMOTIONS_VISUALS,
} from '../types/emotion';

const Options: React.FC = () => {
  const [settings, setSettings] = useState<ExtensionSettings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);
  const [tab, setTab] = useState<'en' | 'vi'>('en');

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const result = await chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }) as any;
      if (result?.payload) setSettings(result.payload);
    } catch (e) {
      console.warn('[Options] Using default settings', e);
    }
  };

  const saveSettings = async (partial: Partial<ExtensionSettings>) => {
    const updated = { ...settings, ...partial };
    setSettings(updated);
    try {
      await chrome.runtime.sendMessage({ type: 'UPDATE_SETTINGS', payload: updated });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      console.error('[Options] Failed to save', e);
    }
  };

  const toggleEmotion = (emotion: EmotionCategory) => {
    const enabled = settings.enabledEmotions.includes(emotion)
      ? settings.enabledEmotions.filter(e => e !== emotion)
      : [...settings.enabledEmotions, emotion];
    saveSettings({ enabledEmotions: enabled });
  };

  return (
    <div className="options-container">
      <header className="options-header">
        <h1>⚙️ Emotion Lens Settings</h1>
        {saved && <span className="saved-badge">✓ Saved</span>}
      </header>

      {/* Model Info */}
      <div className="model-banner">
        <strong>Model:</strong> GoEmotions XLM-RoBERTa + LoRA (28 labels)
        <br />
        <strong>English:</strong> 28 fine-grained emotions
        <br />
        <strong>Vietnamese:</strong> 9 coarse emotions (aggregated from 28)
      </div>

      {/* General Settings */}
      <section className="options-section">
        <h2>General</h2>
        {[
          { key: 'enabled', label: 'Extension Enabled' },
          { key: 'highlightEnabled', label: 'Show Highlights' },
          { key: 'labelsEnabled', label: 'Show Labels' },
          { key: 'toxicityFilterEnabled', label: 'Toxicity Filter' },
          { key: 'localOnly', label: 'Local Only (no backend)' },
        ].map(({ key, label }) => (
          <div key={key} className="setting-row">
            <label>{label}</label>
            <input type="checkbox" checked={(settings as any)[key]}
              onChange={e => saveSettings({ [key]: e.target.checked } as any)} />
          </div>
        ))}
      </section>

      {/* Sensitivity */}
      <section className="options-section">
        <h2>Sensitivity</h2>
        <div className="setting-row">
          <label>Confidence Threshold: {(settings.confidenceThreshold * 100).toFixed(0)}%</label>
          <input type="range" min="0" max="100" value={settings.confidenceThreshold * 100}
            onChange={e => saveSettings({ confidenceThreshold: parseInt(e.target.value) / 100 })} />
        </div>
        <div className="setting-row">
          <label>Sensitivity: {(settings.sensitivity * 100).toFixed(0)}%</label>
          <input type="range" min="0" max="100" value={settings.sensitivity * 100}
            onChange={e => saveSettings({ sensitivity: parseInt(e.target.value) / 100 })} />
        </div>
      </section>

      {/* Emotion Categories */}
      <section className="options-section">
        <h2>Emotion Categories</h2>
        
        {/* Tab Switcher */}
        <div className="tab-bar">
          <button className={`tab-btn ${tab === 'en' ? 'active' : ''}`} onClick={() => setTab('en')}>
            🇬🇧 English · 28 Emotions
          </button>
          <button className={`tab-btn ${tab === 'vi' ? 'active' : ''}`} onClick={() => setTab('vi')}>
            🇻🇳 Vietnamese · 9 Emotions
          </button>
        </div>

        {/* 28 English Emotions */}
        {tab === 'en' && (
          <div className="emotion-grid-28">
            {Object.entries(GOEMOTIONS_28_VISUALS).map(([key, visual]) => (
              <div key={key}
                className={`emotion-card ${settings.enabledEmotions.includes(key as EmotionCategory) ? 'enabled' : ''}`}
                onClick={() => toggleEmotion(key as EmotionCategory)}
                style={{ borderColor: visual.color }}>
                <span className="emotion-icon">{visual.icon}</span>
                <div className="emotion-card-info">
                  <span className="emotion-name">{visual.label}</span>
                  <span className="emotion-group">{visual.group}</span>
                </div>
                <span className="emotion-toggle">
                  {settings.enabledEmotions.includes(key as EmotionCategory) ? '✓' : '✗'}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* 9 Vietnamese Emotions */}
        {tab === 'vi' && (
          <div className="emotion-grid-9">
            {Object.entries(COARSE_EMOTIONS_VISUALS).map(([key, visual]) => (
              <div key={key}
                className={`emotion-card coarse ${settings.enabledEmotions.includes(key as EmotionCategory) ? 'enabled' : ''}`}
                onClick={() => toggleEmotion(key as EmotionCategory)}
                style={{ borderColor: visual.color }}>
                <span className="emotion-icon">{visual.icon}</span>
                <span className="emotion-name">{visual.label}</span>
                <span className="emotion-toggle">
                  {settings.enabledEmotions.includes(key as EmotionCategory) ? '✓' : '✗'}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Reset */}
      <section className="options-section">
        <h2>Reset</h2>
        <button className="reset-btn" onClick={async () => {
          await chrome.runtime.sendMessage({ type: 'UPDATE_SETTINGS', payload: DEFAULT_SETTINGS });
          setSettings(DEFAULT_SETTINGS);
        }}>Reset to Defaults</button>
      </section>
    </div>
  );
};

// Mount
const root = document.createElement('div');
root.id = 'root';
document.body.appendChild(root);
createRoot(root).render(<Options />);