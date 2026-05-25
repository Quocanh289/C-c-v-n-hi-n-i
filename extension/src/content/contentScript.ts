// ====================================================
// Content Script - Core Engine
// Handles DOM observation, text extraction, analysis,
// and visual overlay rendering with 28-label support
// ====================================================

import { emotionClassifier } from '../inference/emotionClassifier';
import { 
  EmotionCategory,
  EmotionResult,
  ExtensionSettings,
  DEFAULT_SETTINGS,
  DEFAULT_EMOTION_VISUALS,
  GOEMOTIONS_28_VISUALS,
  MENTAL_HEALTH_LABELS,
  MENTAL_HEALTH_VISUALS,
  ExtensionMessage,
  MessageType,
  SocialPlatform,
  PLATFORM_SELECTORS,
} from '../types/emotion';

// ====================================================
// Constants
// ====================================================
const CONTENT_SCRIPT_ID = 'emotion-lens';
const OVERLAY_CONTAINER_CLASS = 'emotion-lens-overlay-container';
const STYLESHEET_ID = 'emotion-lens-styles';
const DEBUG_PREFIX = '[EmotionLens]';
const DEBUG_ENABLED = true;

const GENERIC_POST_COMMENT_SELECTOR = [
  'article',
  'div[role="article"]',
  '[data-testid*="post"]',
  '[data-testid*="comment"]',
  'shreddit-post',
  'shreddit-comment',
  'ytd-comment-thread-renderer',
  'div[class*="comment"]',
  'div[class*="post"]',
  'li[class*="comment"]',
].join(', ');

const TARGET_TEXT_SELECTOR = [
  '[data-ad-preview="message"]',
  'div[data-ad-comet-preview="message"]',
  '#content-text',
  'yt-formatted-string#content-text',
  'div[data-testid="tweetText"]',
  '[data-e2e="comment-text"]',
  '[data-e2e="browse-video-desc"]',
  'shreddit-comment [slot="comment"]',
  'shreddit-post [slot="text-body"]',
  'div[data-testid="comment"] p',
  'div.md p',
].join(', ');

const CHAT_EXCLUDE_SELECTOR = [
  '[contenteditable="true"]',
  '[role="textbox"]',
  '[aria-multiline="true"]',
  '[aria-label*="chat" i]',
  '[aria-label*="chats" i]',
  '[aria-label*="conversation" i]',
  '[aria-label*="message" i]',
  '[aria-label*="messenger" i]',
  '[aria-label*="direct message" i]',
  '[aria-label*="inbox" i]',
  '[data-pagelet*="Messenger"]',
  '[data-testid*="chat" i]',
  '[data-testid*="conversation" i]',
  '[data-testid*="dm" i]',
  '[data-testid*="messageDrawer" i]',
  '[data-testid*="DMDrawer" i]',
  '[data-e2e*="chat" i]',
  'div[role="dialog"][aria-label*="chat" i]',
  'div[role="dialog"][aria-label*="conversation" i]',
  'div[role="dialog"][aria-label*="message" i]',
  'div[role="dialog"][aria-label*="messenger" i]',
  'div[role="complementary"][aria-label*="chat" i]',
  'div[role="complementary"][aria-label*="conversation" i]',
  'section[aria-label*="chat" i]',
  'section[aria-label*="conversation" i]',
  'ytd-live-chat-frame',
  'yt-live-chat-app',
  'yt-live-chat-text-message-renderer',
].join(', ');

const TEXT_NODE_EXCLUDE_SELECTOR = [
  `.${OVERLAY_CONTAINER_CLASS}`,
  CHAT_EXCLUDE_SELECTOR,
  'a',
  'button',
  '[role="button"]',
  '[role="link"]',
  'time',
  '[datetime]',
  'header',
  'nav',
  'footer',
  'aside',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
  '[data-testid="User-Name"]',
  '[data-testid="user-name"]',
  '[data-testid="socialContext"]',
  '[data-testid="app-text-transition-container"]',
  '[data-e2e*="username"]',
  '[data-e2e*="like"]',
  '[data-e2e*="comment-count"]',
].join(', ');

const METADATA_TEXT_PATTERNS = [
  /^@\w[\w.]{1,30}$/,
  /^u\/[\w-]+$/i,
  /^r\/[\w-]+$/i,
  /^\d+([,.]\d+)?\s*(k|m)?$/i,
  /^\d+\s*(comments?|replies|likes?|shares?|views?|upvotes?|downvotes?)$/i,
  /^(like|reply|share|follow|following|subscribe|subscribed|view|views|comment|comments)$/i,
  /^(just now|today|yesterday|\d+\s*(s|m|h|d|w|mo|y|sec|secs|min|mins|hr|hrs|day|days|week|weeks|month|months|year|years)\s*ago)$/i,
];

// ====================================================
// State
// ====================================================
let settings: ExtensionSettings = DEFAULT_SETTINGS;
let processedElements = new WeakSet<HTMLElement>();
let processedTexts = new Set<string>();
let analysisQueue: Array<{ element: HTMLElement; text: string }> = [];
let isProcessing = false;
let observer: MutationObserver | null = null;
let scrollObserver: IntersectionObserver | null = null;
let platform: SocialPlatform = SocialPlatform.Unknown;
let batchTimer: ReturnType<typeof setTimeout> | null = null;
let debugCounters = {
  scans: 0,
  candidates: 0,
  queued: 0,
  analyzed: 0,
  overlays: 0,
  filtered: 0,
  errors: 0,
};

function debugLog(message: string, data?: unknown): void {
  if (!DEBUG_ENABLED) return;
  if (data === undefined) {
    console.debug(`${DEBUG_PREFIX} ${message}`);
  } else {
    console.debug(`${DEBUG_PREFIX} ${message}`, data);
  }
}

function normalizeSettings(rawSettings?: Partial<ExtensionSettings>): ExtensionSettings {
  const normalized = { ...DEFAULT_SETTINGS, ...rawSettings } as ExtensionSettings;
  if (!rawSettings?.backendApiUrl || rawSettings.backendApiUrl === 'http://localhost:8000') {
    normalized.backendApiUrl = DEFAULT_SETTINGS.backendApiUrl;
  }
  return normalized;
}

// ====================================================
// Styles Injection - Updated for 28-label colors
// ====================================================
function injectStyles(): void {
  if (document.getElementById(STYLESHEET_ID)) return;
  
  const style = document.createElement('style');
  style.id = STYLESHEET_ID;
  style.textContent = `
    /* ===== Emotion Lens Overlay Styles ===== */
    
    .emotion-lens-overlay-container {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      margin: 0 4px;
      font-size: 11px;
      line-height: 1;
      pointer-events: none;
      user-select: none;
      opacity: 0;
      transition: opacity 0.2s ease-in-out;
      vertical-align: middle;
    }
    
    .emotion-lens-overlay-container.visible {
      opacity: 1;
    }
    
    .emotion-lens-badge {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      padding: 2px 6px;
      border-radius: 8px;
      font-weight: 500;
      font-size: 10px;
      letter-spacing: 0.02em;
      white-space: nowrap;
      pointer-events: auto;
      cursor: default;
      border: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    /* 28-label dynamic colors via inline style */
    .emotion-lens-badge:hover {
      filter: brightness(1.2);
      transform: scale(1.05);
    }
    
    /* Highlight effects */
    .emotion-lens-highlight {
      transition: box-shadow 0.3s ease, background-color 0.3s ease;
      border-radius: 2px;
    }
    
    .emotion-lens-highlight.active {
      box-shadow: 0 0 6px rgba(99, 102, 241, 0.3);
      background-color: rgba(99, 102, 241, 0.04);
    }
    
    @media (prefers-color-scheme: dark) {
      .emotion-lens-badge { border-color: rgba(255, 255, 255, 0.08); }
    }
    
    @media (prefers-color-scheme: light) {
      .emotion-lens-badge { border-color: rgba(0, 0, 0, 0.06); }
    }
  `;
  
  document.head.appendChild(style);
}

// ====================================================
// Platform Detection
// ====================================================
function detectPlatform(): SocialPlatform {
  const hostname = window.location.hostname;
  
  if (hostname.includes('facebook.com')) return SocialPlatform.Facebook;
  if (hostname.includes('youtube.com')) return SocialPlatform.YouTube;
  if (hostname.includes('reddit.com')) return SocialPlatform.Reddit;
  if (hostname.includes('tiktok.com')) return SocialPlatform.TikTok;
  if (hostname.includes('threads.net')) return SocialPlatform.Threads;
  if (hostname.includes('twitter.com')) return SocialPlatform.Twitter;
  if (hostname.includes('x.com')) return SocialPlatform.X;
  
  return SocialPlatform.Unknown;
}

// ====================================================
// Text Extraction
// ====================================================
function hasMetadataAttribute(element: HTMLElement): boolean {
  const attrText = [
    element.getAttribute('aria-label'),
    element.getAttribute('data-testid'),
    element.getAttribute('data-e2e'),
  ].filter(Boolean).join(' ').toLowerCase();

  return /\b(user|username|author|avatar|profile|timestamp|time|date|like|reaction|reply|share|view|count|badge|verified|follow|subscribe|menu|more|vote|score)\b/.test(attrText);
}

function isTextNodeExcluded(parent: HTMLElement): boolean {
  if (isInsideChatSurface(parent)) return true;
  if (parent.closest(TARGET_TEXT_SELECTOR)) return false;
  if (parent.closest(TEXT_NODE_EXCLUDE_SELECTOR)) return true;
  let current: HTMLElement | null = parent;
  while (current && current !== document.body) {
    if (isChatSurface(current)) return true;
    if (current.matches(TARGET_TEXT_SELECTOR)) return false;
    if (hasMetadataAttribute(current)) return true;
    current = current.parentElement;
  }
  return false;
}

function isLikelyMetadataText(text: string): boolean {
  const normalized = text.trim().replace(/\s+/g, ' ');
  if (!normalized) return true;
  if (METADATA_TEXT_PATTERNS.some(pattern => pattern.test(normalized))) return true;
  if (/^[\d\s.,:•·|/+-]+$/.test(normalized)) return true;
  if (normalized.length <= 2) return true;
  return false;
}

function isChatSurface(element: HTMLElement): boolean {
  return element.matches(CHAT_EXCLUDE_SELECTOR);
}

function isInsideChatSurface(element: HTMLElement): boolean {
  return Boolean(element.closest(CHAT_EXCLUDE_SELECTOR));
}

function isAllowedContentElement(element: HTMLElement): boolean {
  if (element.closest(`.${OVERLAY_CONTAINER_CLASS}`)) return false;
  if (isInsideChatSurface(element)) return false;
  if (element.matches(TARGET_TEXT_SELECTOR)) return true;
  if (element.closest(TEXT_NODE_EXCLUDE_SELECTOR)) return false;
  return Boolean(element.closest(GENERIC_POST_COMMENT_SELECTOR));
}

function extractTextFromElement(element: HTMLElement): string {
  if (processedElements.has(element)) return '';
  if (element.closest(`.${OVERLAY_CONTAINER_CLASS}`)) return '';
  if (isInsideChatSurface(element)) return '';
  if (!isAllowedContentElement(element)) return '';
  
  const tagName = element.tagName.toLowerCase();
  if (['script', 'style', 'noscript', 'iframe', 'svg', 'canvas', 'input', 'textarea', 'select'].includes(tagName)) return '';
  
  const texts: string[] = [];
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null as unknown as NodeFilter);
  
  let node: Text | null;
  while (node = walker.nextNode() as Text | null) {
    if (node.parentElement?.closest(`.${OVERLAY_CONTAINER_CLASS}`)) continue;
    const parent = node.parentElement;
    if (parent) {
      if (isTextNodeExcluded(parent)) continue;
      const style = window.getComputedStyle(parent);
      if (style.display === 'none' || style.visibility === 'hidden') continue;
    }
    const text = node.textContent?.trim();
    if (text && !isLikelyMetadataText(text)) texts.push(text);
  }
  
  const extracted = texts.join(' ').replace(/\s+/g, ' ').trim();
  return isLikelyMetadataText(extracted) ? '' : extracted;
}

function findTextElements(): HTMLElement[] {
  const selectors = PLATFORM_SELECTORS[platform] || [];
  const elements: HTMLElement[] = [];
  const seen = new WeakSet<HTMLElement>();

  for (const selector of selectors) {
    const found = document.querySelectorAll<HTMLElement>(selector);
    found.forEach(el => {
      if (!seen.has(el) && !processedElements.has(el) && isAllowedContentElement(el)) {
        seen.add(el);
        elements.push(el);
      }
    });
  }

  // Social platforms change markup often. Keep platform-specific selectors fast,
  // but always fall back to generic post/comment scanning when they miss.
  findGenericTextElements().forEach(el => {
    if (!seen.has(el)) {
      seen.add(el);
      elements.push(el);
    }
  });

  return elements;
}

function findGenericTextElements(): HTMLElement[] {
  const elements: HTMLElement[] = [];
  const candidates = document.querySelectorAll<HTMLElement>(`${TARGET_TEXT_SELECTOR}, ${GENERIC_POST_COMMENT_SELECTOR}`);
  candidates.forEach(el => {
    if (!processedElements.has(el) && isAllowedContentElement(el) && el.textContent && el.textContent.trim().length > 10) elements.push(el);
  });
  return elements;
}

// ====================================================
// Analysis Queue & Batch Processing
// ====================================================
function queueAnalysis(element: HTMLElement, text: string): void {
  if (isInsideChatSurface(element)) return;
  if (!isAllowedContentElement(element)) return;
  if (isLikelyMetadataText(text)) return;
  debugCounters.queued++;
  debugLog('queued text', { mode: settings.activeMode, length: text.length, preview: text.slice(0, 120) });
  analysisQueue.push({ element, text });
  if (batchTimer) clearTimeout(batchTimer);
  batchTimer = setTimeout(processAnalysisQueue, 150);
}

async function processAnalysisQueue(): Promise<void> {
  if (isProcessing || analysisQueue.length === 0) return;
  isProcessing = true;
  
  const batch = analysisQueue.splice(0, 10);
  
  await new Promise<void>((resolve) => {
    requestAnimationFrame(async () => {
      for (const { element, text } of batch) {
        try {
          if (!settings.enabled) continue;
          if (processedTexts.has(text)) continue;
          if (!emotionClassifier.hasContent(text)) continue;
          
          // Check if mode is mental health (English or Vietnamese)
          const isMentalHealthMode = settings.activeMode === 'mental_health_en' || settings.activeMode === 'mental_health_vi';
          const result = isMentalHealthMode
            ? await emotionClassifier.analyzeMentalHealth(text, settings)
            : await emotionClassifier.analyze(text, settings);
          debugCounters.analyzed++;
          debugLog('analysis result', {
            mode: settings.activeMode,
            type: result.analysisType,
            label: result.primaryEmotion,
            confidence: result.confidence,
            language: result.language,
            source: result.source,
            threshold: settings.confidenceThreshold,
          });
          if (!isResultForActiveMode(result)) {
            debugCounters.filtered++;
            debugLog('filtered by active mode', { activeMode: settings.activeMode, resultType: result.analysisType, language: result.language });
            processedTexts.add(text);
            processedElements.add(element);
            continue;
          }
          
          if (result.confidence >= settings.confidenceThreshold) {
            applyVisualOverlay(element, text, result);
          } else {
            debugCounters.filtered++;
            debugLog('filtered by confidence threshold', { confidence: result.confidence, threshold: settings.confidenceThreshold });
          }
          processedTexts.add(text);
          processedElements.add(element);
        } catch (error) {
          debugCounters.errors++;
          console.error('[EmotionLens] Analysis error:', error);
        }
      }
      resolve();
    });
  });
  
  isProcessing = false;
  if (analysisQueue.length > 0) processAnalysisQueue();
}

// ====================================================
// Visual Overlay - Shows the primary label on detected text
// ====================================================
function applyVisualOverlay(element: HTMLElement, text: string, result: EmotionResult): void {
  if (!settings.highlightEnabled && !settings.labelsEnabled) return;
  const isMentalHealth = result.analysisType === 'mental_health';
  
  // Use the 28-label primary emotion if available (English text)
  const emotion28 = result.scores28 ? getTopEmotion28(result.scores28) : null;
  const confidence = result.confidence;
  
  // Get visual config for the detected emotion
  const emotionKey = result.primaryEmotion as EmotionCategory;
  // Check if 28-label result is available
  const has28Label = emotion28 !== null && emotion28 in GOEMOTIONS_28_VISUALS;
  
  let labelText = emotion28 || result.primaryEmotion;
  let icon = '😐';
  let color = '#6b7280';
  let bgColor = 'rgba(107, 114, 128, 0.1)';
  
  if (isMentalHealth && MENTAL_HEALTH_LABELS.includes(result.primaryEmotion as any)) {
    const visual = MENTAL_HEALTH_VISUALS[result.primaryEmotion as keyof typeof MENTAL_HEALTH_VISUALS];
    labelText = visual.label;
    icon = visual.icon;
    color = visual.color;
    bgColor = visual.bgColor;
  } else if (has28Label && emotion28) {
    const v28 = GOEMOTIONS_28_VISUALS[emotion28 as keyof typeof GOEMOTIONS_28_VISUALS];
    if (v28) {
      labelText = v28.label;
      icon = v28.icon;
      color = v28.color;
      bgColor = v28.bgColor;
    }
  } else if (emotionKey in DEFAULT_EMOTION_VISUALS) {
    const v = DEFAULT_EMOTION_VISUALS[emotionKey];
    labelText = v.label;
    icon = v.icon;
    color = v.color;
    bgColor = v.bgColor;
  }
  
  // Check if we should apply toxicity filter
  if (settings.toxicityFilterEnabled && result.toxicityScore > 0.6) {
    // Could add content blurring here
  }
  
  // 1. Apply highlight effect
  if (settings.highlightEnabled) {
    element.classList.add('emotion-lens-highlight', 'active');
    element.style.setProperty('--emotion-glow', `0 0 6px ${color}40`);
  }
  
  // 2. Add emotion badge with 28-label info (always show badge for analyzed text)
  if (settings.labelsEnabled) {
    const existingBadge = element.querySelector(`.${OVERLAY_CONTAINER_CLASS}`);
    if (existingBadge) return;
    
    const overlayContainer = document.createElement('span');
    overlayContainer.className = `${OVERLAY_CONTAINER_CLASS}`;
    
    const badge = document.createElement('span');
    badge.className = `emotion-lens-badge`;
    badge.style.background = bgColor;
    badge.style.color = color;
    badge.style.borderColor = `${color}40`;
    badge.title = isMentalHealth
      ? `${labelText}: ${(confidence * 100).toFixed(0)}% | ${result.language === 'vi' ? 'VI->EN mental health model' : 'Mental health model'}${result.severityLabel ? ` | ${result.severityLabel}` : ''}`
      : `${labelText}: ${(confidence * 100).toFixed(0)}% | ${result.language === 'vi' ? 'VI->EN GoEmotions 28-label model' : 'GoEmotions 28-label model'}`;
    badge.textContent = `${icon} ${labelText}`.trim();
    overlayContainer.appendChild(badge);
    try {
      element.insertAdjacentElement('afterend', overlayContainer);
    } catch {
      element.appendChild(overlayContainer);
    }
    debugCounters.overlays++;
    debugLog('overlay inserted', { label: labelText, confidence, totalOverlays: debugCounters.overlays });
    
    requestAnimationFrame(() => {
      overlayContainer.classList.add('visible');
    });
  }
  
  // Send result to background
  chrome.runtime.sendMessage({
    type: MessageType.ANALYSIS_RESULT,
    payload: { text, result },
  }).catch(() => {});
}

/**
 * Get the highest scoring 28-label emotion.
 */
function getTopEmotion28(scores28: Record<string, number>): string | null {
  let topEmotion: string | null = null;
  let topScore = 0;
  
  for (const [label, score] of Object.entries(scores28)) {
    if (label !== 'neutral' && score > topScore) {
      topScore = score;
      topEmotion = label;
    }
  }
  
  return topEmotion;
}

function isResultForActiveMode(result: EmotionResult): boolean {
  if (settings.activeMode === 'mental_health_en' || settings.activeMode === 'mental_health_vi') {
    return result.analysisType === 'mental_health';
  }
  if (settings.activeMode === 'emotion_en') {
    return result.analysisType !== 'mental_health' && result.language === 'en';
  }
  // emotion_vi - accept vietnamese or mixed language, non-mental-health results
  return result.analysisType !== 'mental_health' && (result.language === 'vi' || result.language === 'mixed');
}

// ====================================================
// DOM Observation
// ====================================================
function initMutationObserver(): void {
  observer = new MutationObserver((mutations) => {
    if (!settings.enabled) return;
    
    let hasNewNodes = false;
    
    for (const mutation of mutations) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        hasNewNodes = true;
        
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            const element = node as HTMLElement;
            
            if (isTextElement(element) && isAllowedContentElement(element)) {
              const text = extractTextFromElement(element);
              if (text && !processedTexts.has(text)) queueAnalysis(element, text);
            }
            
            const textElements = element.querySelectorAll<HTMLElement>(
              `${TARGET_TEXT_SELECTOR}, ${GENERIC_POST_COMMENT_SELECTOR}`
            );
            textElements.forEach(child => {
              if (!processedElements.has(child) && isAllowedContentElement(child)) {
                const childText = extractTextFromElement(child);
                if (childText && !processedTexts.has(childText)) queueAnalysis(child, childText);
              }
            });
          }
        }
      }
      
      if (mutation.type === 'characterData') {
        const target = mutation.target as Text;
        const parent = target.parentElement;
        if (parent && !processedElements.has(parent) && isAllowedContentElement(parent)) {
          const text = extractTextFromElement(parent);
          if (text && !processedTexts.has(text)) queueAnalysis(parent, text);
        }
      }
    }
    
    if (hasNewNodes) scheduleFullScan();
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
    characterDataOldValue: false,
  });
}

function isTextElement(element: HTMLElement): boolean {
  const tag = element.tagName.toLowerCase();
  if (['script', 'style', 'noscript', 'iframe', 'svg', 'button', 'a'].includes(tag)) return false;
  if (!isAllowedContentElement(element)) return false;
  
  const text = element.textContent?.trim();
  if (!text || text.length < 5) return false;
  if (isLikelyMetadataText(text)) return false;
  
  return /[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]{2,}/i.test(text);
}

// ====================================================
// Intersection Observer
// ====================================================
function initScrollObserver(): void {
  scrollObserver = new IntersectionObserver(
    (entries) => {
      if (!settings.enabled) return;
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const element = entry.target as HTMLElement;
          if (!processedElements.has(element)) {
            const text = extractTextFromElement(element);
            if (text && !processedTexts.has(text)) queueAnalysis(element, text);
          }
          scrollObserver?.unobserve(element);
        }
      });
    },
    { rootMargin: '200px 0px', threshold: 0.1 }
  );
}

let scanTimeout: ReturnType<typeof setTimeout> | null = null;

function scheduleFullScan(): void {
  if (scanTimeout) clearTimeout(scanTimeout);
  scanTimeout = setTimeout(() => performFullScan(), 1000);
}

function performFullScan(): void {
  if (!settings.enabled) return;
  const elements = findTextElements();
  debugCounters.scans++;
  debugCounters.candidates += elements.length;
  debugLog('scan', {
    platform,
    mode: settings.activeMode,
    elements: elements.length,
    counters: debugCounters,
  });
  
  for (const element of elements) {
    if (!processedElements.has(element)) {
      const text = extractTextFromElement(element);
      if (text && !processedTexts.has(text)) {
        if (scrollObserver) {
          scrollObserver.observe(element);
        } else {
          queueAnalysis(element, text);
        }
      }
    }
  }
}

// ====================================================
// Settings Listener
// ====================================================
function listenForSettings(): void {
  chrome.runtime.onMessage.addListener((message: unknown) => {
    const msg = message as ExtensionMessage;
    
    switch (msg.type) {
      case MessageType.SETTINGS_UPDATED:
        settings = normalizeSettings(msg.payload as Partial<ExtensionSettings>);
        if (!settings.enabled) removeAllOverlays();
        if (settings.enabled) {
          removeAllOverlays();
          scheduleFullScan();
        }
        break;
        
      case MessageType.PROCESS_ELEMENT:
        const { element } = msg.payload as { element: HTMLElement };
        const text = extractTextFromElement(element);
        if (text) queueAnalysis(element, text);
        break;
    }
    
    return false;
  });

  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== 'sync' || !changes.emotionLensSettings?.newValue) return;
    settings = normalizeSettings(changes.emotionLensSettings.newValue as Partial<ExtensionSettings>);
    debugLog('settings changed via storage', settings);
    if (!settings.enabled) {
      removeAllOverlays();
      return;
    }
    removeAllOverlays();
    scheduleFullScan();
  });
}

// ====================================================
// Cleanup
// ====================================================
function removeAllOverlays(): void {
  document.querySelectorAll(`.${OVERLAY_CONTAINER_CLASS}`).forEach(el => el.remove());
  document.querySelectorAll<HTMLElement>('.emotion-lens-highlight').forEach(el => {
    el.classList.remove('emotion-lens-highlight', 'active');
    el.style.removeProperty('--emotion-glow');
  });
  processedElements = new WeakSet();
  processedTexts.clear();
  analysisQueue = [];
}

function cleanup(): void {
  if (observer) observer.disconnect();
  if (scrollObserver) scrollObserver.disconnect();
  if (batchTimer) clearTimeout(batchTimer);
  if (scanTimeout) clearTimeout(scanTimeout);
  removeAllOverlays();
}

// ====================================================
// Initialization
// ====================================================
async function initialize(): Promise<void> {
  try {
    platform = detectPlatform();
    console.log(`[EmotionLens] Detected platform: ${platform}`);
    
    const result = await chrome.storage.sync.get(['emotionLensSettings']);
    settings = normalizeSettings(result.emotionLensSettings as Partial<ExtensionSettings> | undefined);
    debugLog('settings loaded', settings);
    
    if (!settings.enabled) {
      console.log('[EmotionLens] Extension is disabled in settings');
      return;
    }
    
    injectStyles();
    
    emotionClassifier.initialize().catch(err => {
      console.warn('[EmotionLens] Classifier initialization warning:', err);
    });
    
    initMutationObserver();
    initScrollObserver();
    listenForSettings();
    performFullScan();
    
    setInterval(() => {
      if (settings.enabled) performFullScan();
    }, 5000);
    
    console.log('[EmotionLens] Successfully initialized - showing 28-label emotions');
    
  } catch (error) {
    console.error('[EmotionLens] Initialization failed:', error);
  }
}

export { cleanup };

// Start the extension
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
