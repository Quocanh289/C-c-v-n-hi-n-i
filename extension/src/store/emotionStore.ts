// ====================================================
// Zustand Store for Emotion Detection State
// Central state management for the entire extension
// ====================================================

import { create } from 'zustand';
import { 
  EmotionCategory, 
  EmotionResult, 
  ExtensionSettings, 
  ExtensionStats,
  DEFAULT_SETTINGS,
  AnalyzedElement,
  ExtensionMessage,
  MessageType,
} from '../types/emotion';

interface EmotionState {
  // Settings
  settings: ExtensionSettings;
  isSettingsLoaded: boolean;
  
  // Analysis results
  analyzedElements: Map<string, AnalyzedElement>;
  processedTexts: Set<string>;
  
  // Stats
  stats: ExtensionStats;
  
  // Connection status
  isConnected: boolean;
  lastError: string | null;
  
  // Actions
  loadSettings: () => Promise<void>;
  updateSettings: (partial: Partial<ExtensionSettings>) => Promise<void>;
  resetSettings: () => Promise<void>;
  
  addAnalysisResult: (elementId: string, element: HTMLElement, text: string, result: EmotionResult) => void;
  getElementResult: (elementId: string) => AnalyzedElement | undefined;
  hasProcessed: (text: string) => boolean;
  
  updateStats: (result: EmotionResult) => void;
  resetStats: () => void;
  getStats: () => ExtensionStats;
  
  clearCache: () => void;
  setLastError: (error: string | null) => void;
}

export const useEmotionStore = create<EmotionState>((set, get) => ({
  settings: DEFAULT_SETTINGS,
  isSettingsLoaded: false,
  analyzedElements: new Map(),
  processedTexts: new Set(),
  stats: {
    totalAnalyzed: 0,
    emotionsDetected: {
      [EmotionCategory.Joy]: 0,
      [EmotionCategory.Anger]: 0,
      [EmotionCategory.Sadness]: 0,
      [EmotionCategory.Anxiety]: 0,
      [EmotionCategory.Fear]: 0,
      [EmotionCategory.Surprise]: 0,
      [EmotionCategory.Neutral]: 0,
      [EmotionCategory.Toxic]: 0,
      [EmotionCategory.Sarcastic]: 0,
    },
    avgConfidence: 0,
    avgInferenceTime: 0,
    cacheSize: 0,
    localInferences: 0,
    backendInferences: 0,
  },
  isConnected: false,
  lastError: null,

  loadSettings: async () => {
    try {
      const result = await chrome.storage.sync.get(['emotionLensSettings']);
      if (result.emotionLensSettings) {
        set({ 
          settings: { ...DEFAULT_SETTINGS, ...result.emotionLensSettings },
          isSettingsLoaded: true,
        });
      } else {
        set({ isSettingsLoaded: true });
      }
    } catch (error) {
      console.error('[EmotionStore] Failed to load settings:', error);
      set({ isSettingsLoaded: true });
    }
  },

  updateSettings: async (partial: Partial<ExtensionSettings>) => {
    const currentSettings = get().settings;
    const newSettings = { ...currentSettings, ...partial };
    
    try {
      await chrome.storage.sync.set({ emotionLensSettings: newSettings });
      set({ settings: newSettings });
      
      // Notify content script of settings change
      const message: ExtensionMessage = {
        type: MessageType.SETTINGS_UPDATED,
        payload: newSettings,
      };
      chrome.runtime.sendMessage(message).catch(() => {});
      
    } catch (error) {
      console.error('[EmotionStore] Failed to save settings:', error);
    }
  },

  resetSettings: async () => {
    await get().updateSettings(DEFAULT_SETTINGS);
  },

  addAnalysisResult: (elementId, element, text, result) => {
    const state = get();
    const newElement: AnalyzedElement = {
      id: elementId,
      element,
      text,
      result,
      analyzedAt: Date.now(),
      overlayVisible: false,
    };
    
    const newMap = new Map(state.analyzedElements);
    newMap.set(elementId, newElement);
    
    const newProcessed = new Set(state.processedTexts);
    newProcessed.add(text);
    
    // Enforce cache size limit
    if (newMap.size > state.settings.maxCacheSize) {
      const oldestKeys = Array.from(newMap.keys())
        .slice(0, newMap.size - state.settings.maxCacheSize);
      oldestKeys.forEach(key => {
        newMap.delete(key);
        newProcessed.delete(newMap.get(key)?.text || '');
      });
    }
    
    set({
      analyzedElements: newMap,
      processedTexts: newProcessed,
      stats: {
        ...state.stats,
        cacheSize: newMap.size,
      },
    });
    
    get().updateStats(result);
  },

  getElementResult: (elementId) => {
    return get().analyzedElements.get(elementId);
  },

  hasProcessed: (text) => {
    return get().processedTexts.has(text);
  },

  updateStats: (result) => {
    const state = get();
    const newEmotionsDetected = { ...state.stats.emotionsDetected };
    newEmotionsDetected[result.primaryEmotion] = 
      (newEmotionsDetected[result.primaryEmotion] || 0) + 1;
    
    const totalAnalyzed = state.stats.totalAnalyzed + 1;
    const newAvgConfidence = 
      (state.stats.avgConfidence * state.stats.totalAnalyzed + result.confidence) / totalAnalyzed;
    const newAvgInferenceTime = 
      (state.stats.avgInferenceTime * state.stats.totalAnalyzed + result.inferenceTimeMs) / totalAnalyzed;
    
    set({
      stats: {
        ...state.stats,
        totalAnalyzed,
        emotionsDetected: newEmotionsDetected,
        avgConfidence: newAvgConfidence,
        avgInferenceTime: newAvgInferenceTime,
        localInferences: result.source === 'local' 
          ? state.stats.localInferences + 1 
          : state.stats.localInferences,
        backendInferences: result.source === 'backend'
          ? state.stats.backendInferences + 1
          : state.stats.backendInferences,
      },
    });
  },

  resetStats: () => {
    set({
      stats: {
        totalAnalyzed: 0,
        emotionsDetected: {
          [EmotionCategory.Joy]: 0,
          [EmotionCategory.Anger]: 0,
          [EmotionCategory.Sadness]: 0,
          [EmotionCategory.Anxiety]: 0,
          [EmotionCategory.Fear]: 0,
          [EmotionCategory.Surprise]: 0,
          [EmotionCategory.Neutral]: 0,
          [EmotionCategory.Toxic]: 0,
          [EmotionCategory.Sarcastic]: 0,
        },
        avgConfidence: 0,
        avgInferenceTime: 0,
        cacheSize: 0,
        localInferences: 0,
        backendInferences: 0,
      },
      analyzedElements: new Map(),
      processedTexts: new Set(),
    });
  },

  getStats: () => {
    return get().stats;
  },

  clearCache: () => {
    set({
      analyzedElements: new Map(),
      processedTexts: new Set(),
      stats: {
        ...get().stats,
        cacheSize: 0,
      },
    });
  },

  setLastError: (error) => {
    set({ lastError: error });
  },
}));