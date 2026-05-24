// ====================================================
// Background Service Worker
// Handles extension lifecycle, message routing,
// stats aggregation, and cache management
// ====================================================

import {
  EmotionResult,
  ExtensionMessage,
  MessageType,
  ExtensionStats,
  ExtensionSettings,
  DEFAULT_SETTINGS,
  EmotionCategory,
} from '../types/emotion';

// ====================================================
// State
// ====================================================
let stats: ExtensionStats = {
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
};

let activeTabId: number | null = null;

let settings: ExtensionSettings = DEFAULT_SETTINGS;

// ====================================================
// Initialization
// ====================================================

/**
 * Initialize the background service worker.
 * Loads settings and sets up message listeners.
 */
async function initialize(): Promise<void> {
  try {
    // Load settings
    const result = await chrome.storage.sync.get(['emotionLensSettings']);
    if (result.emotionLensSettings) {
      settings = { ...DEFAULT_SETTINGS, ...result.emotionLensSettings } as ExtensionSettings;
    }

    // Load stats
    const statsResult = await chrome.storage.local.get(['emotionLensStats']);
    if (statsResult.emotionLensStats) {
      stats = statsResult.emotionLensStats as ExtensionStats;
    }

    // Track active tab
    chrome.tabs.query({ active: true, currentWindow: true }).then(tabs => {
      if (tabs.length > 0) {
        activeTabId = tabs[0].id ?? null;
        updateBadge();
      }
    });

    console.log('[EmotionLens] Background service worker initialized');
  } catch (error) {
    console.error('[EmotionLens] Background initialization failed:', error);
  }
}

// ====================================================
// Badge Update
// ====================================================

/**
 * Update the extension badge with current stats.
 */
function updateBadge(): void {
  if (!activeTabId) return;

  const count = stats.totalAnalyzed;
  if (count > 0) {
    chrome.action.setBadgeText({
      text: count > 999 ? '999+' : count.toString(),
      tabId: activeTabId,
    });
    chrome.action.setBadgeBackgroundColor({
      color: '#6366f1',
      tabId: activeTabId,
    });
  } else {
    chrome.action.setBadgeText({
      text: '',
      tabId: activeTabId,
    });
  }
}

// ====================================================
// Stats Management
// ====================================================

function recordInference(result: EmotionResult): void {
  stats.totalAnalyzed++;
  stats.emotionsDetected[result.primaryEmotion] =
    (stats.emotionsDetected[result.primaryEmotion] || 0) + 1;

  // Rolling average for confidence
  stats.avgConfidence =
    (stats.avgConfidence * (stats.totalAnalyzed - 1) + result.confidence) / stats.totalAnalyzed;

  // Rolling average for inference time
  stats.avgInferenceTime =
    (stats.avgInferenceTime * (stats.totalAnalyzed - 1) + result.inferenceTimeMs) / stats.totalAnalyzed;

  if (result.source === 'local') {
    stats.localInferences++;
  } else if (result.source === 'backend') {
    stats.backendInferences++;
  }

  stats.cacheSize = stats.totalAnalyzed;

  // Persist stats periodically (throttled)
  persistStats();

  updateBadge();
}

let statsPersistTimer: ReturnType<typeof setTimeout> | null = null;

function persistStats(): void {
  if (statsPersistTimer) clearTimeout(statsPersistTimer);
  statsPersistTimer = setTimeout(() => {
    chrome.storage.local.set({ emotionLensStats: stats }).catch(err => {
      console.error('[EmotionLens] Failed to persist stats:', err);
    });
  }, 2000);
}

function resetStats(): void {
  stats = {
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
  };

  chrome.storage.local.set({ emotionLensStats: stats }).catch(err => {
    console.error('[EmotionLens] Failed to reset stats:', err);
  });

  updateBadge();
}

// ====================================================
// Message Handling
// ====================================================

chrome.runtime.onMessage.addListener(
  (message: unknown, sender: chrome.runtime.MessageSender, sendResponse: (response?: unknown) => void) => {
    const msg = message as ExtensionMessage;

    switch (msg.type) {
      case MessageType.ANALYSIS_RESULT: {
        const { result } = msg.payload as { text: string; result: EmotionResult };
        recordInference(result);
        break;
      }

      case MessageType.GET_STATS: {
        sendResponse({ type: MessageType.STATS_RESULT, payload: stats });
        return true; // Keep channel open for async response
      }

      case MessageType.GET_SETTINGS: {
        sendResponse({ type: MessageType.SETTINGS_RESULT, payload: settings });
        return true;
      }

      case MessageType.UPDATE_SETTINGS: {
        const newSettings = msg.payload as Partial<ExtensionSettings>;
        settings = { ...settings, ...newSettings };
        chrome.storage.sync.set({ emotionLensSettings: settings }).catch(err => {
          console.error('[EmotionLens] Failed to save settings:', err);
        });

        // Forward to content script
        if (activeTabId) {
          chrome.tabs.sendMessage(activeTabId, {
            type: MessageType.SETTINGS_UPDATED,
            payload: settings,
          }).catch(() => {
            // Tab might not have content script loaded
          });
        }

        sendResponse({ type: MessageType.SETTINGS_RESULT, payload: settings });
        return true;
      }

      case MessageType.RESET_CACHE: {
        resetStats();
        sendResponse({ payload: { success: true } });
        return true;
      }
    }

    return false;
  }
);

// ====================================================
// Tab Tracking
// ====================================================

chrome.tabs.onActivated.addListener((activeInfo) => {
  activeTabId = activeInfo.tabId;
  updateBadge();
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'complete') {
    // Reset badge for refreshed tabs
    if (tabId === activeTabId) {
      updateBadge();
    }
  }
});

// ====================================================
// Extension Lifecycle
// ====================================================



chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    // Set default settings on first install
    chrome.storage.sync.set({ emotionLensSettings: DEFAULT_SETTINGS }).catch(err => {
      console.error('[EmotionLens] Failed to set default settings:', err);
    });
    chrome.storage.local.set({ emotionLensStats: stats }).catch(err => {
      console.error('[EmotionLens] Failed to set initial stats:', err);
    });

    // Open options page on install
    chrome.tabs.create({ url: 'dist/options/index.html' }).catch(() => {});
  }

  if (details.reason === 'update') {
    console.log('[EmotionLens] Extension updated');
  }
});
// extension/src/background/background.ts

// 1. Chạy logic khởi tạo khi tiện ích vừa được cài đặt hoặc cập nhật
chrome.runtime.onInstalled.addListener(async () => {
  // Sinh mã định danh ẩn danh (UUID) nếu chưa có
  const storage = await chrome.storage.local.get(['userId']);
  if (!storage.userId) {
    const uuid = crypto.randomUUID(); // Hàm có sẵn của trình duyệt sinh UUID an toàn
    await chrome.storage.local.set({ userId: uuid });
    console.log("Đã khởi tạo UUID cho người dùng mới:", uuid);
  }

  // Khởi tạo Context Menu
  chrome.contextMenus.create({
    id: "save-to-emotion-lens",
    title: "Save to Emotion Lens",
    contexts: ["selection"] // XÁC ĐỊNH: Chỉ hiện khi người dùng quét bôi đen văn bản
  });
});

// 2. Xử lý sự kiện khi Context Menu được click
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "save-to-emotion-lens" && info.selectionText) {
    
    // Lấy UUID ẩn danh từ Storage
    const storage = await chrome.storage.local.get('userId');
    const userId = storage.userId;

    try {
      // Gửi POST request có chứa UID và đoạn văn bản tới server Backend
      const response = await fetch("http://localhost:8000/api/save", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          uid: userId,
          text: info.selectionText,
          sourceUrl: tab?.url || ""
        })
      });

      if (response.ok) {
        console.log("Đã lưu chuỗi văn bản thành công về server!");
        // Có thể bổ sung tính năng tạo thông báo notification trên màn hình ở đây
      } else {
        console.error("Lỗi khi lưu dữ liệu về Backend:", response.status);
      }
    } catch (error) {
      console.error("Lỗi rớt mạng hoặc hệ thống:", error);
    }
  }
});
// Initialize
initialize();