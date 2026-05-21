// ====================================================
// Side Panel - Detailed Emotion Analyzer
// Shows full 28/9 emotion analysis from trained model
// ====================================================

import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  GOEMOTIONS_28_VISUALS,
  COARSE_EMOTIONS_VISUALS,
  EMOTION_28_TO_9_MAP,
} from '../types/emotion';

const SidePanel: React.FC = () => {
  const [tab, setTab] = useState<'en' | 'vi'>('en');

  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, -apple-system, sans-serif', color: '#1f2937' }}>
      <style>{`
        body { margin: 0; background: #f9fafb; }
        @media (prefers-color-scheme: dark) {
          body { background: #111827; }
        }
      `}</style>
      
      {/* Header */}
      <h1 style={{ fontSize: 18, marginBottom: 4, fontWeight: 700 }}>
        📊 Emotion Lens
      </h1>
      <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 16 }}>
        GoEmotions 28-label model · Detects {tab === 'en' ? '28 fine-grained' : '9 coarse'} emotions
      </p>

      {/* Tab Switcher */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button onClick={() => setTab('en')} style={{
          flex: 1, padding: '10px 16px', borderRadius: 8, border: '1px solid',
          borderColor: tab === 'en' ? '#6366f1' : '#d1d5db',
          background: tab === 'en' ? '#eef2ff' : 'white',
          color: tab === 'en' ? '#4f46e5' : '#6b7280',
          fontSize: 13, fontWeight: 600, cursor: 'pointer',
        }}>
          🇬🇧 28 Emotions (EN)
        </button>
        <button onClick={() => setTab('vi')} style={{
          flex: 1, padding: '10px 16px', borderRadius: 8, border: '1px solid',
          borderColor: tab === 'vi' ? '#f97316' : '#d1d5db',
          background: tab === 'vi' ? '#fff7ed' : 'white',
          color: tab === 'vi' ? '#ea580c' : '#6b7280',
          fontSize: 13, fontWeight: 600, cursor: 'pointer',
        }}>
          🇻🇳 9 Emotions (VI)
        </button>
      </div>

      {/* English: 28 Emotions */}
      {tab === 'en' && (
        <div>
          <p style={{ fontSize: 11, color: '#9ca3af', marginBottom: 8, lineHeight: 1.4 }}>
            These 28 fine-grained emotions are from the GoEmotions dataset.
            They are detected by the trained XLM-RoBERTa model.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6 }}>
            {Object.entries(GOEMOTIONS_28_VISUALS).map(([key, visual]) => (
              <div key={key} style={{
                background: 'white', borderRadius: 8, padding: '10px 12px',
                border: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', gap: 10,
              }}>
                <span style={{ fontSize: 18 }}>{visual.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 600 }}>{visual.label}</div>
                  <div style={{ fontSize: 10, color: '#9ca3af', marginTop: 1 }}>
                    <span style={{ color: visual.color }}>●</span> {visual.group}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Vietnamese: 9 Coarse Emotions */}
      {tab === 'vi' && (
        <div>
          <p style={{ fontSize: 11, color: '#f97316', marginBottom: 8, padding: '8px 12px', background: '#fff7ed', borderRadius: 8, border: '1px solid #fed7aa', lineHeight: 1.4 }}>
            <strong>Note:</strong> Vietnamese emotions are aggregated from the 28-label model.
            Dedicated Vietnamese training data coming soon.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
            {Object.entries(COARSE_EMOTIONS_VISUALS).map(([key, visual]) => {
              const subEmotions = Object.entries(GOEMOTIONS_28_VISUALS)
                .filter(([, v]) => v.group === key)
                .map(([k]) => k);
              return (
                <div key={key} style={{
                  background: 'white', borderRadius: 10, padding: '12px',
                  border: '2px solid #e5e7eb', borderLeftColor: visual.color,
                  borderLeftWidth: 4,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <span style={{ fontSize: 20 }}>{visual.icon}</span>
                    <span style={{ fontSize: 14, fontWeight: 700 }}>{visual.label}</span>
                  </div>
                  <div style={{ fontSize: 10, color: '#9ca3af', lineHeight: 1.5 }}>
                    Includes: {subEmotions.join(', ')}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 20, padding: '12px 0', borderTop: '1px solid #e5e7eb', fontSize: 10, color: '#9ca3af', textAlign: 'center' }}>
        Emotion Lens · GoEmotions XLM-RoBERTa + LoRA
        <br />
        28 EN / 9 VI · Auto language detection
      </div>
    </div>
  );
};

// Mount
const root = document.createElement('div');
root.id = 'root';
document.body.appendChild(root);
createRoot(root).render(<SidePanel />);