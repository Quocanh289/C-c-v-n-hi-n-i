// ====================================================
// Options/Settings Page (stub)
// Full settings UI for the Emotion Lens extension
// ====================================================

import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { EmotionCategory, ExtensionSettings, DEFAULT_SETTINGS, DEFAULT_EMOTION_VISUALS } from '../types/emotion';

const Options: React.FC = () => {
  const [settings, setSettings] = useState<ExtensionSettings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const result = await chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }) as { payload?: ExtensionSettings };
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

      <section className="options-section">
        <h2>General</h2>
        <div className="setting-row">
          <label>Extension Enabled</label>
          <input type="checkbox" checked={settings.enabled} onChange={e => saveSettings({ enabled: e.target.checked })} />
        </div>
        <div className="setting-row">
          <label>Show Highlights</label>
          <input type="checkbox" checked={settings.highlightEnabled} onChange={e => saveSettings({ highlightEnabled: e.target.checked })} />
        </div>
        <div className="setting-row">
          <label>Show Labels</label>
          <input type="checkbox" checked={settings.labelsEnabled} onChange={e => saveSettings({ labelsEnabled: e.target.checked })} />
        </div>
        <div className="setting-row">
          <label>Toxicity Filter</label>
          <input type="checkbox" checked={settings.toxicityFilterEnabled} onChange={e => saveSettings({ toxicityFilterEnabled: e.target.checked })} />
        </div>
        <div className="setting-row">
          <label>Local Only</label>
          <input type="checkbox" checked={settings.localOnly} onChange={e => saveSettings({ localOnly: e.target.checked })} />
        </div>
      </section>

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

      <section className="options-section">
        <h2>Emotion Categories</h2>
        <div className="emotion-grid">
          {Object.entries(DEFAULT_EMOTION_VISUALS).map(([key, visual]) => (
            <div key={key} className={`emotion-card ${settings.enabledEmotions.includes(key as EmotionCategory) ? 'enabled' : ''}`}
              onClick={() => toggleEmotion(key as EmotionCategory)}
              style={{ borderColor: visual.color }}>
              <span className="emotion-icon">{visual.icon}</span>
              <span className="emotion-name">{visual.label}</span>
              <span className="emotion-toggle">{settings.enabledEmotions.includes(key as EmotionCategory) ? '✓' : '✗'}</span>
            </div>
          ))}
        </div>
      </section>

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