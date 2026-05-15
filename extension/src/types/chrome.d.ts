// ====================================================
// Chrome Extension API Type Declarations
// Provides type safety for Chrome Extension APIs
// ====================================================

declare namespace chrome {
  namespace storage {
    interface StorageArea {
      get(keys: string[]): Promise<Record<string, unknown>>;
      set(items: Record<string, unknown>): Promise<void>;
      remove(keys: string[]): Promise<void>;
      clear(): Promise<void>;
    }

    const sync: StorageArea;
    const local: StorageArea;
  }

  namespace runtime {
    function sendMessage(message: unknown): Promise<unknown>;
    function sendMessage(extensionId: string | undefined, message: unknown, options?: unknown): Promise<unknown>;

    interface MessageSender {
      tab?: chrome.tabs.Tab;
      id?: string;
      url?: string;
      origin?: string;
    }

    interface RuntimeMessageEvent {
      addListener(
        callback: (
          message: unknown,
          sender: MessageSender,
          sendResponse: (response?: unknown) => void
        ) => void | boolean
      ): void;
      removeListener(callback: (...args: unknown[]) => void): void;
    }

    interface InstalledDetails {
      reason: 'install' | 'update' | 'chrome_update' | 'shared_module_update';
      previousVersion?: string;
      id?: string;
    }

    const onMessage: RuntimeMessageEvent;
    const onInstalled: { addListener(callback: (details: InstalledDetails) => void): void };
    const onConnect: { addListener(callback: (port: unknown) => void): void };
    function openOptionsPage(callback?: () => void): void;

    interface Port {
      name: string;
      disconnect: () => void;
      postMessage: (message: unknown) => void;
      onMessage: { addListener: (callback: (message: unknown) => void) => void };
      onDisconnect: { addListener: (callback: (port: Port) => void) => void };
    }

    function connect(connectInfo?: { name?: string; includeTlsChannelId?: boolean }): Port;
  }

  namespace tabs {
    interface Tab {
      id?: number;
      url?: string;
      title?: string;
      active?: boolean;
      windowId?: number;
      index?: number;
    }

    interface ActiveInfo {
      tabId: number;
      windowId?: number;
    }

    interface TabChangeInfo {
      status?: string;
      url?: string;
      title?: string;
    }

    const onActivated: {
      addListener(callback: (activeInfo: ActiveInfo) => void): void;
      removeListener(callback: (activeInfo: ActiveInfo) => void): void;
    };

    const onUpdated: {
      addListener(
        callback: (tabId: number, changeInfo: TabChangeInfo, tab: Tab) => void
      ): void;
      removeListener(
        callback: (tabId: number, changeInfo: TabChangeInfo, tab: Tab) => void
      ): void;
    };

    function query(queryInfo: { active?: boolean; currentWindow?: boolean }): Promise<Tab[]>;
    function sendMessage(tabId: number, message: unknown): Promise<unknown>;
    function create(createProperties: { url?: string; active?: boolean }): Promise<Tab>;
  }

  namespace action {
    function setBadgeText(details: { text: string; tabId?: number }): void;
    function setBadgeBackgroundColor(details: { color: string; tabId?: number }): void;
    function setTitle(details: { title: string; tabId?: number }): void;
  }

  namespace sidePanel {
    interface SidePanelOptions {
      enabled?: boolean;
      path?: string;
    }
    function setOptions(options: SidePanelOptions): Promise<void>;
    function open(options?: { tabId?: number }): Promise<void>;
  }

  namespace scripting {
    interface InjectionTarget {
      tabId: number;
      frameIds?: number[];
    }
    interface ScriptInjection {
      files?: string[];
      target?: InjectionTarget;
      func?: (...args: unknown[]) => void;
    }
    function executeScript(injection: ScriptInjection): Promise<unknown[]>;
    function insertCSS(injection: { files: string[]; target?: InjectionTarget }): Promise<void>;
  }

  namespace i18n {
    function getMessage(messageName: string, substitutions?: string | string[]): string;
    function getUILanguage(): string;
  }
}

interface Window {
  chrome: typeof chrome;
}