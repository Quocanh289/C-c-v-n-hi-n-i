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
  COARSE_EMOTIONS_VISUALS,
  MENTAL_HEALTH_VISUALS,
} from '../types/emotion';

type PanelTab = 'en' | 'vi' | 'mh';

const modeToTab = (mode: DetectionMode): PanelTab => {
  if (mode === 'mental_health_en') return 'mh';
  if (mode === 'emotion_vi') return 'vi';
  return 'en';
};

const tabToMode = (tab: PanelTab): DetectionMode => {
  if (tab === 'mh') return 'mental_health_en';
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
    <button onClick={() => selectTab(name)} style={{
      flex: 1,
      padding: '8px 10px',
      borderRadius: 8,
      border: `1px solid ${tab === name ? color : '#d1d5db'}`,
      background: tab === name ? '#f8fafc' : 'white',
      color: tab === name ? color : '#6b7280',
      fontSize: 12,
      fontWeight: 700,
      cursor: 'pointer',
    }}>
      {label}
    </button>
  );

  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1f2937' }}>
      <style>{`body { margin: 0; background: #f9fafb; }`}</style>
      <h1 style={{ fontSize: 18, marginBottom: 4, fontWeight: 700 }}>Emotion Lens</h1>
      <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 16 }}>
        Only one mode scans at a time. Text is taken from posts and comments.
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 16 }}>
        {tabButton('en', 'EN Emotion', '#4f46e5')}
        {tabButton('vi', 'VI Emotion', '#ea580c')}
        {tabButton('mh', 'EN Mental', '#7c3aed')}
      </div>

      {tab === 'en' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6 }}>
          {Object.entries(GOEMOTIONS_28_VISUALS).map(([key, visual]) => (
            <div key={key} style={{ background: 'white', borderRadius: 8, padding: '10px 12px', border: '1px solid #e5e7eb', borderLeft: `4px solid ${visual.color}` }}>
              <div style={{ fontSize: 12, fontWeight: 700 }}>{visual.label}</div>
              <div style={{ fontSize: 10, color: '#6b7280', marginTop: 2 }}>{visual.group}</div>
            </div>
          ))}
        </div>
      )}

      {tab === 'vi' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {Object.entries(COARSE_EMOTIONS_VISUALS).map(([key, visual]) => (
            <div key={key} style={{ background: 'white', borderRadius: 8, padding: '12px', border: '1px solid #e5e7eb', borderLeft: `4px solid ${visual.color}` }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>{visual.label}</div>
            </div>
          ))}
        </div>
      )}

      {tab === 'mh' && (
        <div>
          <p style={{ fontSize: 11, color: '#7c3aed', marginBottom: 8, lineHeight: 1.4 }}>
            Uses the backend mental-health endpoint with DeBERTa-v3 + LoRA from best_model.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6 }}>
            {Object.entries(MENTAL_HEALTH_VISUALS).map(([key, visual]) => (
              <div key={key} style={{ background: 'white', borderRadius: 8, padding: '10px 12px', border: '1px solid #e5e7eb', borderLeft: `4px solid ${visual.color}` }}>
                <div style={{ fontSize: 12, fontWeight: 700 }}>{visual.label}</div>
                <div style={{ fontSize: 10, color: '#6b7280', marginTop: 2 }}>{visual.severityLabel}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginTop: 16, padding: '12px 0', borderTop: '1px solid #e5e7eb', fontSize: 10, color: '#6b7280', textAlign: 'center' }}>
        Active mode: {tab === 'mh' ? 'English Mental Health' : tab === 'vi' ? 'Vietnamese Emotion' : 'English Emotion'}
      </div>
    </div>
  );
};

const root = document.createElement('div');
root.id = 'root';
document.body.appendChild(root);
createRoot(root).render(<SidePanel />);
