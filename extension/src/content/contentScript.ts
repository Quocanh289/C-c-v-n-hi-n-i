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
function extractTextFromElement(element: HTMLElement): string {
  if (processedElements.has(element)) return '';
  if (element.closest(`.${OVERLAY_CONTAINER_CLASS}`)) return '';
  
  const tagName = element.tagName.toLowerCase();
  if (['script', 'style', 'noscript', 'iframe', 'svg', 'canvas'].includes(tagName)) return '';
  
  const texts: string[] = [];
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null as unknown as NodeFilter);
  
  let node: Text | null;
  while (node = walker.nextNode() as Text | null) {
    if (node.parentElement?.closest(`.${OVERLAY_CONTAINER_CLASS}`)) continue;
    const parent = node.parentElement;
    if (parent) {
      const style = window.getComputedStyle(parent);
      if (style.display === 'none' || style.visibility === 'hidden') continue;
    }
    const text = node.textContent?.trim();
    if (text && text.length > 1) texts.push(text);
  }
  
  return texts.join(' ').trim();
}

function findTextElements(): HTMLElement[] {
  const selectors = PLATFORM_SELECTORS[platform] || [];
  if (selectors.length === 0) return findGenericTextElements();
  
  const elements: HTMLElement[] = [];
  for (const selector of selectors) {
    const found = document.querySelectorAll<HTMLElement>(selector);
    found.forEach(el => {
      if (!processedElements.has(el)) elements.push(el);
    });
  }
  return elements;
}

function findGenericTextElements(): HTMLElement[] {
  const elements: HTMLElement[] = [];
  const candidates = document.querySelectorAll<HTMLElement>(
    'p, span, div[role="article"], div[class*="comment"], div[class*="post"], div[class*="message"], li[class*="comment"]'
  );
  candidates.forEach(el => {
    if (!processedElements.has(el) && el.textContent && el.textContent.trim().length > 10) elements.push(el);
  });
  return elements;
}

// ====================================================
// Analysis Queue & Batch Processing
// ====================================================
function queueAnalysis(element: HTMLElement, text: string): void {
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
          
          processedTexts.add(text);
          processedElements.add(element);
          
          const result = await emotionClassifier.analyze(text, settings);
          
          if (result.confidence >= settings.confidenceThreshold) {
            applyVisualOverlay(element, text, result);
          }
        } catch (error) {
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
// Visual Overlay - Shows 28-label badges on detected text
// ====================================================
function applyVisualOverlay(element: HTMLElement, text: string, result: EmotionResult): void {
  if (!settings.highlightEnabled && !settings.labelsEnabled) return;
  
  // Use the 28-label primary emotion if available (English text)
  const emotion28 = result.scores28 ? getTopEmotion28(result.scores28) : null;
  const primaryLabel = emotion28 || result.primaryEmotion;
  const confidence = result.confidence;
  
  // Get visual config for the detected emotion
  const emotionKey = result.primaryEmotion as EmotionCategory;
  // Check if 28-label result is available
  const has28Label = emotion28 !== null && emotion28 in GOEMOTIONS_28_VISUALS;
  
  let labelText = emotion28 || result.primaryEmotion;
  let icon = '😐';
  let color = '#6b7280';
  let bgColor = 'rgba(107, 114, 128, 0.1)';
  
  // Use 28-label visuals if available (English text with fine-grained labels)
  if (has28Label && emotion28) {
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
    badge.title = `${labelText}: ${(confidence * 100).toFixed(0)}% | ${result.language === 'en' ? '28-label' : '9-label'} model`;
    badge.textContent = `${icon} ${labelText}`;
    
    overlayContainer.appendChild(badge);
    element.insertAdjacentElement('afterend', overlayContainer);
    
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
            
            if (isTextElement(element)) {
              const text = extractTextFromElement(element);
              if (text && !processedTexts.has(text)) queueAnalysis(element, text);
            }
            
            const textElements = element.querySelectorAll<HTMLElement>(
              'p, span, div[data-testid], [dir="auto"], #content-text, [data-e2e="comment-text"]'
            );
            textElements.forEach(child => {
              if (!processedElements.has(child)) {
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
        if (parent && !processedElements.has(parent)) {
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
  if (['script', 'style', 'noscript', 'iframe', 'svg'].includes(tag)) return false;
  
  const text = element.textContent?.trim();
  if (!text || text.length < 5) return false;
  
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
        settings = msg.payload as ExtensionSettings;
        if (!settings.enabled) removeAllOverlays();
        break;
        
      case MessageType.PROCESS_ELEMENT:
        const { element } = msg.payload as { element: HTMLElement };
        const text = extractTextFromElement(element);
        if (text) queueAnalysis(element, text);
        break;
    }
    
    return false;
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
    if (result.emotionLensSettings) {
      settings = { ...DEFAULT_SETTINGS, ...result.emotionLensSettings } as ExtensionSettings;
    }
    
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