// ====================================================
// Side Panel - mode control and label reference
// ====================================================

import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  DEFAULT_SETTINGS,
  DetectionMode,
  ExtensionSettings,
  GOEMOTIONS_28_VISUALS,
  MENTAL_HEALTH_VISUALS,
} from '../types/emotion';

type PanelTab = 'en' | 'vi' | 'mh' | 'mh_vi';

const modeToTab = (mode: DetectionMode): PanelTab => {
  if (mode === 'mental_health_en') return 'mh';
  if (mode === 'mental_health_vi') return 'mh_vi';
  if (mode === 'emotion_vi') return 'vi';
  return 'en';
};

const tabToMode = (tab: PanelTab): DetectionMode => {
  if (tab === 'mh') return 'mental_health_en';
  if (tab === 'mh_vi') return 'mental_health_vi';
  if (tab === 'vi') return 'emotion_vi';
  return 'emotion_en';
};

const SidePanel: React.FC = () => {
  const [settings, setSettings] = useState<ExtensionSettings>(DEFAULT_SETTINGS);
  const [tab, setTab] = useState<PanelTab>('en');

  useEffect(() => {
    chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }).then((result: any) => {
      if (result?.payload) {
        setSettings(result.payload);
        setTab(modeToTab(result.payload.activeMode || DEFAULT_SETTINGS.activeMode));
      }
    }).catch(() => {});
  }, []);

  const selectTab = async (nextTab: PanelTab) => {
    const updated = { ...settings, activeMode: tabToMode(nextTab) };
    setTab(nextTab);
    setSettings(updated);
    await chrome.runtime.sendMessage({ type: 'UPDATE_SETTINGS', payload: updated });
  };

  const tabButton = (name: PanelTab, label: string, color: string) => (
    <button
      className={`panel-tab ${tab === name ? 'active' : ''}`}
      style={{ borderColor: tab === name ? color : undefined }}
      onClick={() => selectTab(name)}
    >
      {label}
    </button>
  );

  const emotionCards = (
    <div className="label-grid">
      {Object.entries(GOEMOTIONS_28_VISUALS).map(([key, visual]) => (
        <div key={key} className="label-card" style={{ borderLeftColor: visual.color }}>
          <div className="label-name">{visual.label}</div>
          <div className="label-group">{visual.group}</div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="panel-shell">
      <header className="panel-header">
        <div>
          <h1 className="panel-title">Emotion Lens</h1>
          <p className="panel-copy">
            Select one scan mode. Badges show the primary result plus secondary signal chips.
          </p>
        </div>
        <span className="mode-pill">v2.1</span>
      </header>

      <div className="panel-tabs">
        {tabButton('en', 'EN Emotion', '#4f46e5')}
        {tabButton('vi', 'VI Emotion', '#ea580c')}
        {tabButton('mh', 'EN Mental', '#7c3aed')}
        {tabButton('mh_vi', 'VI Mental', '#db2777')}
      </div>

      {tab === 'en' && emotionCards}

      {tab === 'vi' && (
        <div>
          <p className="panel-note">
            Vietnamese text is translated to English before the 28-label emotion model runs.
          </p>
          {emotionCards}
        </div>
      )}

      {(tab === 'mh' || tab === 'mh_vi') && (
        <div>
          <p className="panel-note">
            {tab === 'mh'
              ? 'Uses backend DeBERTa-v3 + LoRA best_model checkpoint.'
              : 'Vietnamese text is translated to English before the DeBERTa-v3 + LoRA mental health model runs.'}
          </p>
          <div className="label-grid">
            {Object.entries(MENTAL_HEALTH_VISUALS).map(([key, visual]) => (
              <div key={key} className="label-card" style={{ borderLeftColor: visual.color }}>
                <div className="label-name">{visual.label}</div>
                <div className="label-group">{visual.severityLabel}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="panel-footer">
        Active mode: {tab === 'en'
          ? 'English Emotion (28-label)'
          : tab === 'vi'
            ? 'Vietnamese Emotion (VI to EN to 28)'
            : tab === 'mh'
              ? 'English Mental Health'
              : 'Vietnamese Mental Health (VI to EN)'}
      </div>
    </div>
  );
};

const root = document.createElement('div');
root.id = 'root';
document.body.appendChild(root);
createRoot(root).render(<SidePanel />);
