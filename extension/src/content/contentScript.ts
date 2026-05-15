// ====================================================
// Content Script - Core Engine
// Handles DOM observation, text extraction, analysis,
// and visual overlay rendering for social media platforms
// ====================================================

import { emotionClassifier } from '../inference/emotionClassifier';
import { 
  EmotionCategory,
  EmotionResult,
  ExtensionSettings,
  DEFAULT_SETTINGS,
  DEFAULT_EMOTION_VISUALS,
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
// Styles Injection
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
      gap: 2px;
      padding: 1px 5px;
      border-radius: 8px;
      font-weight: 500;
      font-size: 10px;
      letter-spacing: 0.02em;
      white-space: nowrap;
      pointer-events: auto;
      cursor: default;
    }
    
    .emotion-lens-badge.joy { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
    .emotion-lens-badge.anger { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .emotion-lens-badge.sadness { background: rgba(59, 130, 246, 0.15); color: #3b82f6; }
    .emotion-lens-badge.anxiety { background: rgba(249, 115, 22, 0.15); color: #f97316; }
    .emotion-lens-badge.fear { background: rgba(249, 115, 22, 0.15); color: #f97316; }
    .emotion-lens-badge.surprise { background: rgba(168, 85, 247, 0.15); color: #a855f7; }
    .emotion-lens-badge.neutral { background: rgba(107, 114, 128, 0.1); color: #6b7280; }
    .emotion-lens-badge.toxic { background: rgba(220, 38, 38, 0.15); color: #dc2626; }
    .emotion-lens-badge.sarcastic { background: rgba(217, 70, 239, 0.15); color: #d946ef; }
    
    /* Highlight effects */
    .emotion-lens-highlight {
      transition: box-shadow 0.3s ease, background-color 0.3s ease;
      border-radius: 2px;
    }
    
    .emotion-lens-highlight.joy {
      box-shadow: 0 0 6px rgba(34, 197, 94, 0.3);
      background-color: rgba(34, 197, 94, 0.04);
    }
    
    .emotion-lens-highlight.anger {
      box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
      background-color: rgba(239, 68, 68, 0.04);
    }
    
    .emotion-lens-highlight.sadness {
      box-shadow: 0 0 6px rgba(59, 130, 246, 0.3);
      background-color: rgba(59, 130, 246, 0.04);
    }
    
    .emotion-lens-highlight.anxiety {
      box-shadow: 0 0 6px rgba(249, 115, 22, 0.35);
      background-color: rgba(249, 115, 22, 0.04);
    }
    
    .emotion-lens-highlight.fear {
      box-shadow: 0 0 6px rgba(249, 115, 22, 0.35);
      background-color: rgba(249, 115, 22, 0.04);
    }
    
    .emotion-lens-highlight.surprise {
      box-shadow: 0 0 6px rgba(168, 85, 247, 0.3);
      background-color: rgba(168, 85, 247, 0.04);
    }
    
    .emotion-lens-highlight.toxic {
      box-shadow: 0 0 10px rgba(220, 38, 38, 0.5);
      background-color: rgba(220, 38, 38, 0.05);
    }
    
    .emotion-lens-highlight.sarcastic {
      box-shadow: 0 0 6px rgba(217, 70, 239, 0.3);
      background-color: rgba(217, 70, 239, 0.04);
    }
    
    /* Badge tooltip on hover */
    .emotion-lens-badge:hover {
      filter: brightness(1.2);
      transform: scale(1.05);
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      .emotion-lens-badge { border: 1px solid rgba(255, 255, 255, 0.08); }
    }
    
    @media (prefers-color-scheme: light) {
      .emotion-lens-badge { border: 1px solid rgba(0, 0, 0, 0.06); }
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

/**
 * Extract visible text content from DOM elements.
 * Skips scripts, styles, and hidden elements.
 */
function extractTextFromElement(element: HTMLElement): string {
  // Skip if already processed
  if (processedElements.has(element)) return '';
  
  // Skip if inside our own overlay
  if (element.closest(`.${OVERLAY_CONTAINER_CLASS}`)) return '';
  
  // Skip non-text elements
  const tagName = element.tagName.toLowerCase();
  if (['script', 'style', 'noscript', 'iframe', 'svg', 'canvas'].includes(tagName)) return '';
  
  // Get all text nodes, filter meaningful content
  const texts: string[] = [];
  
  const walker = document.createTreeWalker(
    element,
    NodeFilter.SHOW_TEXT,
    null as unknown as NodeFilter
  );

  // Filter text nodes manually
  const filterTextNode = (node: Text): boolean => {
    // Skip empty text nodes
    if (!node.textContent || node.textContent.trim().length === 0) {
      return false;
    }
    // Skip nodes inside our overlay
    if (node.parentElement?.closest(`.${OVERLAY_CONTAINER_CLASS}`)) {
      return false;
    }
    // Skip hidden elements
    const parent = node.parentElement;
    if (parent) {
      const style = window.getComputedStyle(parent);
      if (style.display === 'none' || style.visibility === 'hidden') {
        return false;
      }
    }
    return true;
  };
  
  let node: Text | null;
  while (node = walker.nextNode() as Text | null) {
    const text = node.textContent?.trim();
    if (text && text.length > 1) {
      texts.push(text);
    }
  }
  
  return texts.join(' ').trim();
}

/**
 * Find text-containing elements on the page for the current platform.
 */
function findTextElements(): HTMLElement[] {
  const selectors = PLATFORM_SELECTORS[platform] || [];
  
  // If no platform-specific selectors, use generic text containers
  if (selectors.length === 0) {
    return findGenericTextElements();
  }
  
  const elements: HTMLElement[] = [];
  for (const selector of selectors) {
    const found = document.querySelectorAll<HTMLElement>(selector);
    found.forEach(el => {
      if (!processedElements.has(el)) {
        elements.push(el);
      }
    });
  }
  
  return elements;
}

/**
 * Generic text element detection for unknown platforms.
 */
function findGenericTextElements(): HTMLElement[] {
  const elements: HTMLElement[] = [];
  const candidates = document.querySelectorAll<HTMLElement>(
    'p, span, div[role="article"], div[class*="comment"], div[class*="post"], div[class*="message"], li[class*="comment"]'
  );
  
  candidates.forEach(el => {
    if (!processedElements.has(el) && el.textContent && el.textContent.trim().length > 10) {
      elements.push(el);
    }
  });
  
  return elements;
}

// ====================================================
// Analysis Queue & Batch Processing
// ====================================================

/**
 * Add elements to the analysis queue and trigger batch processing.
 */
function queueAnalysis(element: HTMLElement, text: string): void {
  analysisQueue.push({ element, text });
  
  // Debounced batch processing - accumulates entries then processes in batch
  if (batchTimer) clearTimeout(batchTimer);
  batchTimer = setTimeout(processAnalysisQueue, 150); // 150ms debounce
}

/**
 * Process analysis queue in batches.
 * Uses RequestAnimationFrame for non-blocking UI performance.
 */
async function processAnalysisQueue(): Promise<void> {
  if (isProcessing || analysisQueue.length === 0) return;
  
  isProcessing = true;
  
  // Take a batch of up to 10 items
  const batch = analysisQueue.splice(0, 10);
  
  // Process batch asynchronously
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
  
  // Process next batch if queue still has items
  if (analysisQueue.length > 0) {
    processAnalysisQueue();
  }
}

// ====================================================
// Visual Overlay Application
// ====================================================

/**
 * Apply visual emotion overlay to a DOM element.
 * Combines highlight effect + emotion badge.
 */
function applyVisualOverlay(element: HTMLElement, text: string, result: EmotionResult): void {
  if (!settings.highlightEnabled && !settings.labelsEnabled) return;
  
  const emotion = result.primaryEmotion;
  if (emotion === EmotionCategory.Neutral && !settings.labelsEnabled) return;
  
  const visual = DEFAULT_EMOTION_VISUALS[emotion];
  
  // Check if we should apply toxicity filter
  if (settings.toxicityFilterEnabled && result.toxicityScore > 0.6) {
    // Could add content blurring here
  }
  
  // 1. Apply highlight effect
  if (settings.highlightEnabled && visual.color) {
    element.classList.add('emotion-lens-highlight', emotion);
    element.style.setProperty('--emotion-glow', visual.glowEffect);
  }
  
  // 2. Add emotion badge
  if (settings.labelsEnabled && emotion !== EmotionCategory.Neutral) {
    // Check if badge already exists
    const existingBadge = element.querySelector(`.${OVERLAY_CONTAINER_CLASS}`);
    if (existingBadge) return;
    
    const overlayContainer = document.createElement('span');
    overlayContainer.className = `${OVERLAY_CONTAINER_CLASS}`;
    
    const badge = document.createElement('span');
    badge.className = `emotion-lens-badge ${emotion}`;
    badge.title = `Confidence: ${(result.confidence * 100).toFixed(0)}% | ${result.language}`;
    badge.textContent = `${visual.icon} ${visual.label}`;
    
    overlayContainer.appendChild(badge);
    
    // Insert badge after the element
    element.insertAdjacentElement('afterend', overlayContainer);
    
    // Fade in
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

// ====================================================
// DOM Observation
// ====================================================

/**
 * Initialize MutationObserver to watch for dynamically loaded content.
 * This is critical for infinite scrolling feeds.
 */
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
            
            // If the added node is itself a text element
            if (isTextElement(element)) {
              const text = extractTextFromElement(element);
              if (text && !processedTexts.has(text)) {
                queueAnalysis(element, text);
              }
            }
            
            // Check children
            const textElements = element.querySelectorAll<HTMLElement>(
              'p, span, div[data-testid], [dir="auto"], #content-text, [data-e2e="comment-text"]'
            );
            
            textElements.forEach(child => {
              if (!processedElements.has(child)) {
                const childText = extractTextFromElement(child);
                if (childText && !processedTexts.has(childText)) {
                  queueAnalysis(child, childText);
                }
              }
            });
          }
        }
      }
      
      // Watch for text content changes
      if (mutation.type === 'characterData') {
        const target = mutation.target as Text;
        const parent = target.parentElement;
        if (parent && !processedElements.has(parent)) {
          const text = extractTextFromElement(parent);
          if (text && !processedTexts.has(text)) {
            queueAnalysis(parent, text);
          }
        }
      }
    }
    
    if (hasNewNodes) {
      // Also scan for any text elements we might have missed
      scheduleFullScan();
    }
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
    characterDataOldValue: false,
  });
}

/**
 * Check if an element is likely a text-containing element.
 */
function isTextElement(element: HTMLElement): boolean {
  const tag = element.tagName.toLowerCase();
  if (['script', 'style', 'noscript', 'iframe', 'svg'].includes(tag)) return false;
  
  const text = element.textContent?.trim();
  if (!text || text.length < 5) return false;
  
  // Must contain actual words (not just whitespace/symbols)
  return /[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]{2,}/i.test(text);
}

// ====================================================
// Intersection Observer for Scroll Optimization
// ====================================================

/**
 * Initialize IntersectionObserver to only analyze visible elements.
 * This prevents analyzing off-screen content, saving battery and CPU.
 */
function initScrollObserver(): void {
  scrollObserver = new IntersectionObserver(
    (entries) => {
      if (!settings.enabled) return;
      
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const element = entry.target as HTMLElement;
          if (!processedElements.has(element)) {
            const text = extractTextFromElement(element);
            if (text && !processedTexts.has(text)) {
              queueAnalysis(element, text);
            }
          }
          // Unobserve after first appearance to avoid re-processing
          scrollObserver?.unobserve(element);
        }
      });
    },
    {
      rootMargin: '200px 0px', // Preload 200px before element enters viewport
      threshold: 0.1,
    }
  );
}

/**
 * Schedule a full page scan for new text elements.
 * Useful for catching missed elements on dynamic feeds.
 */
let scanTimeout: ReturnType<typeof setTimeout> | null = null;

function scheduleFullScan(): void {
  if (scanTimeout) clearTimeout(scanTimeout);
  scanTimeout = setTimeout(() => {
    performFullScan();
  }, 1000); // 1 second debounce
}

function performFullScan(): void {
  if (!settings.enabled) return;
  
  const elements = findTextElements();
  
  for (const element of elements) {
    if (!processedElements.has(element)) {
      const text = extractTextFromElement(element);
      if (text && !processedTexts.has(text)) {
        // Use IntersectionObserver if available, otherwise queue directly
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

/**
 * Listen for settings updates from background script.
 */
function listenForSettings(): void {
  chrome.runtime.onMessage.addListener((message: unknown) => {
    const msg = message as ExtensionMessage;
    
    switch (msg.type) {
      case MessageType.SETTINGS_UPDATED:
        settings = msg.payload as ExtensionSettings;
        console.log('[EmotionLens] Settings updated:', settings);
        
        if (!settings.enabled) {
          removeAllOverlays();
        }
        break;
        
      case MessageType.PROCESS_ELEMENT:
        const { element } = msg.payload as { element: HTMLElement };
        const text = extractTextFromElement(element);
        if (text) {
          queueAnalysis(element, text);
        }
        break;
    }
    
    return false; // Don't keep message channel open
  });
}

// ====================================================
// Cleanup
// ====================================================

/**
 * Remove all emotion overlays from the page.
 */
function removeAllOverlays(): void {
  document.querySelectorAll(`.${OVERLAY_CONTAINER_CLASS}`).forEach(el => el.remove());
  document.querySelectorAll<HTMLElement>('.emotion-lens-highlight').forEach(el => {
    el.classList.remove(...el.classList.toString().match(/emotion-lens-highlight\s+\S+/g) || []);
    el.style.removeProperty('--emotion-glow');
  });
  
  // Clear processed tracking
  processedElements = new WeakSet();
  processedTexts.clear();
  analysisQueue = [];
}

/**
 * Cleanup function for extension reload/disable.
 */
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
    // Detect platform
    platform = detectPlatform();
    console.log(`[EmotionLens] Detected platform: ${platform}`);
    
    // Load settings
    const result = await chrome.storage.sync.get(['emotionLensSettings']);
    if (result.emotionLensSettings) {
      settings = { ...DEFAULT_SETTINGS, ...result.emotionLensSettings } as ExtensionSettings;
    }
    
    if (!settings.enabled) {
      console.log('[EmotionLens] Extension is disabled in settings');
      return;
    }
    
    // Inject styles
    injectStyles();
    
    // Initialize classifier (lazy load - will init on first analysis)
    emotionClassifier.initialize().catch(err => {
      console.warn('[EmotionLens] Classifier initialization warning:', err);
    });
    
    // Initialize observers
    initMutationObserver();
    initScrollObserver();
    
    // Listen for settings changes
    listenForSettings();
    
    // Initial scan of existing page content
    performFullScan();
    
    // Re-scan periodically for platforms with highly dynamic content
    setInterval(() => {
      if (settings.enabled) {
        performFullScan();
      }
    }, 5000); // Every 5 seconds
    
    console.log('[EmotionLens] Successfully initialized');
    
  } catch (error) {
    console.error('[EmotionLens] Initialization failed:', error);
  }
}

// Export for cleanup if needed (e.g., HMR in development)
export { cleanup };

// Start the extension
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}