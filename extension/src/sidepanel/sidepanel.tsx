import React from 'react';
import { createRoot } from 'react-dom/client';

const SidePanel: React.FC = () => {
  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif' }}>
      <h1 style={{ fontSize: 18, marginBottom: 12 }}>📊 Emotion Lens</h1>
      <p style={{ color: '#6b7280', fontSize: 13 }}>Detailed analytics coming soon.</p>
    </div>
  );
};

const root = document.createElement('div');
root.id = 'root';
document.body.appendChild(root);
createRoot(root).render(<SidePanel />);