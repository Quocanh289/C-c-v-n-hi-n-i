// ====================================================
// Core Types for Emotion Detection System
// ====================================================

/** Emotion categories detected by the system */
export enum EmotionCategory {
  Joy = 'joy',
  Anger = 'anger',
  Sadness = 'sadness',
  Anxiety = 'anxiety',
  Fear = 'fear',
  Surprise = 'surprise',
  Neutral = 'neutral',
  Toxic = 'toxic',
  Sarcastic = 'sarcastic',
}

/** Confidence scores for each emotion category */
export type EmotionScores = Record<EmotionCategory, number>;

/** Complete emotion analysis result */
export interface EmotionResult {
  /** Primary detected emotion */
  primaryEmotion: EmotionCategory;
  /** All emotion scores (confidence 0-1) */
  scores: EmotionScores;
  /** Confidence level of the primary emotion */
  confidence: number;
  /** Detected toxicity level (0-1) */
  toxicityScore: number;
  /** Detected sarcasm likelihood (0-1) */
  sarcasmScore: number;
  /** Detected language */
  language: 'vi' | 'en' | 'mixed';
  /** Whether this was inferred locally or via backend */
  source: 'local' | 'backend' | 'cache';
  /** Inference time in ms */
  inferenceTimeMs: number;
}

/** Visual configuration for emotion display */
export interface EmotionVisual {
  /** Display label */
  label: string;
  /** Emoji icon */
  icon: string;
  /** Primary color (hex) */
  color: string;
  /** Background color (hex with alpha) */
  bgColor: string;
  /** Border glow effect CSS */
  glowEffect: string;
  /** Whether to show this emotion in UI */
  enabled: boolean;
}

/** Text element reference with its analysis */
export interface AnalyzedElement {
  /** Unique element ID */
  id: string;
  /** The DOM element */
  element: HTMLElement;
  /** Original text content */
  text: string;
  /** Emotion analysis result */
  result: EmotionResult;
  /** Timestamp of analysis */
  analyzedAt: number;
  /** Whether overlay is currently visible */
  overlayVisible: boolean;
}

/** Settings for the extension */
export interface ExtensionSettings {
  /** Master toggle */
  enabled: boolean;
  /** Enable/disable text highlighting */
  highlightEnabled: boolean;
  /** Enable/disable overlay labels/icons */
  labelsEnabled: boolean;
  /** Enable toxic content filtering */
  toxicityFilterEnabled: boolean;
  /** Minimum confidence threshold (0-1) */
  confidenceThreshold: number;
  /** Sensitivity level (0-1) */
  sensitivity: number;
  /** Theme preference */
  theme: 'light' | 'dark' | 'system';
  /** Custom colors per emotion category */
  customColors: Partial<Record<EmotionCategory, string>>;
  /** Toggle individual emotion categories */
  enabledEmotions: EmotionCategory[];
  /** Backend API URL (optional, for fallback) */
  backendApiUrl: string;
  /** Use local inference only */
  localOnly: boolean;
  /** Cache results locally */
  cacheEnabled: boolean;
  /** Maximum cache size */
  maxCacheSize: number;
}

/** Default emotion visuals mapping */
export const DEFAULT_EMOTION_VISUALS: Record<EmotionCategory, EmotionVisual> = {
  [EmotionCategory.Joy]: {
    label: 'Positive',
    icon: '✨',
    color: '#22c55e',
    bgColor: 'rgba(34, 197, 94, 0.1)',
    glowEffect: '0 0 8px rgba(34, 197, 94, 0.5)',
    enabled: true,
  },
  [EmotionCategory.Anger]: {
    label: 'Angry',
    icon: '😡',
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.1)',
    glowEffect: '0 0 8px rgba(239, 68, 68, 0.5)',
    enabled: true,
  },
  [EmotionCategory.Sadness]: {
    label: 'Sad',
    icon: '😢',
    color: '#3b82f6',
    bgColor: 'rgba(59, 130, 246, 0.1)',
    glowEffect: '0 0 8px rgba(59, 130, 246, 0.4)',
    enabled: true,
  },
  [EmotionCategory.Anxiety]: {
    label: 'Anxious',
    icon: '😰',
    color: '#f97316',
    bgColor: 'rgba(249, 115, 22, 0.1)',
    glowEffect: '0 0 8px rgba(249, 115, 22, 0.5)',
    enabled: true,
  },
  [EmotionCategory.Fear]: {
    label: 'Fearful',
    icon: '😨',
    color: '#f97316',
    bgColor: 'rgba(249, 115, 22, 0.1)',
    glowEffect: '0 0 8px rgba(249, 115, 22, 0.5)',
    enabled: true,
  },
  [EmotionCategory.Surprise]: {
    label: 'Surprised',
    icon: '😲',
    color: '#a855f7',
    bgColor: 'rgba(168, 85, 247, 0.1)',
    glowEffect: '0 0 8px rgba(168, 85, 247, 0.5)',
    enabled: true,
  },
  [EmotionCategory.Neutral]: {
    label: 'Neutral',
    icon: '😐',
    color: '#6b7280',
    bgColor: 'transparent',
    glowEffect: 'none',
    enabled: true,
  },
  [EmotionCategory.Toxic]: {
    label: 'Toxic',
    icon: '☠',
    color: '#dc2626',
    bgColor: 'rgba(220, 38, 38, 0.08)',
    glowEffect: '0 0 10px rgba(220, 38, 38, 0.6)',
    enabled: true,
  },
  [EmotionCategory.Sarcastic]: {
    label: 'Sarcasm',
    icon: '🎭',
    color: '#d946ef',
    bgColor: 'rgba(217, 70, 239, 0.1)',
    glowEffect: '0 0 8px rgba(217, 70, 239, 0.5)',
    enabled: true,
  },
};

/** Default settings */
export const DEFAULT_SETTINGS: ExtensionSettings = {
  enabled: true,
  highlightEnabled: true,
  labelsEnabled: true,
  toxicityFilterEnabled: false,
  confidenceThreshold: 0.35,
  sensitivity: 0.5,
  theme: 'system',
  customColors: {},
  enabledEmotions: Object.values(EmotionCategory),
  backendApiUrl: 'http://localhost:8000',
  localOnly: true,
  cacheEnabled: true,
  maxCacheSize: 500,
};

/** Message protocol between extension components */
export interface ExtensionMessage {
  type: MessageType;
  payload?: unknown;
}

export enum MessageType {
  // Content -> Background
  ANALYSIS_RESULT = 'ANALYSIS_RESULT',
  ELEMENT_UPDATED = 'ELEMENT_UPDATED',
  
  // Background -> Content
  PROCESS_ELEMENT = 'PROCESS_ELEMENT',
  SETTINGS_UPDATED = 'SETTINGS_UPDATED',
  
  // Popup/Sidepanel -> Background
  GET_STATS = 'GET_STATS',
  GET_SETTINGS = 'GET_SETTINGS',
  UPDATE_SETTINGS = 'UPDATE_SETTINGS',
  RESET_CACHE = 'RESET_CACHE',
  
  // Background -> Popup/Sidepanel
  STATS_RESULT = 'STATS_RESULT',
  SETTINGS_RESULT = 'SETTINGS_RESULT',
}

export interface ExtensionStats {
  totalAnalyzed: number;
  emotionsDetected: Record<EmotionCategory, number>;
  avgConfidence: number;
  avgInferenceTime: number;
  cacheSize: number;
  localInferences: number;
  backendInferences: number;
}

/** Social media platform identifiers */
export enum SocialPlatform {
  Facebook = 'facebook',
  YouTube = 'youtube',
  Reddit = 'reddit',
  TikTok = 'tiktok',
  Threads = 'threads',
  Twitter = 'twitter',
  X = 'x',
  Unknown = 'unknown',
}

/** Selectors for finding text content per platform */
export const PLATFORM_SELECTORS: Record<SocialPlatform, string[]> = {
  [SocialPlatform.Facebook]: [
    '[data-ad-preview="message"]',
    'div[data-ad-comet-preview="message"]',
    'span[dir="auto"]',
    'div[role="article"] div[style*="text-align"]',
  ],
  [SocialPlatform.YouTube]: [
    '#content-text',
    'yt-formatted-string#content-text',
    '#comment-content',
    'ytd-comment-thread-renderer #content-text',
  ],
  [SocialPlatform.Reddit]: [
    'div[data-testid="comment"] p',
    'shreddit-comment div.md p',
    'div.md p',
    'h3',
  ],
  [SocialPlatform.TikTok]: [
    'div[data-e2e="comment-text"]',
    'span[data-e2e="comment-username"]',
  ],
  [SocialPlatform.Threads]: [
    'div[data-pressable-container="true"] span',
  ],
  [SocialPlatform.Twitter]: [
    'div[data-testid="tweetText"]',
    'article div[lang] span',
  ],
  [SocialPlatform.X]: [
    'div[data-testid="tweetText"]',
    'article div[lang] span',
  ],
  [SocialPlatform.Unknown]: [],
};