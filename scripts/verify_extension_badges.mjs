import { readFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const chromePath = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const userDataDir = `${process.cwd()}/.tmp/chrome-badge-verify`;
const port = 9223;

const contentScript = await readFile('extension/dist/content.js', 'utf8');

const chrome = spawn(chromePath, [
  '--headless=new',
  '--remote-allow-origins=*',
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${userDataDir}`,
  '--disable-gpu',
  'about:blank',
], { stdio: 'ignore' });

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

async function getWebSocketDebuggerUrl() {
  for (let i = 0; i < 80; i += 1) {
    try {
      const tabs = await json(`http://127.0.0.1:${port}/json`);
      const page = tabs.find((tab) => tab.type === 'page');
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch {
      await delay(100);
    }
  }
  throw new Error('Chrome DevTools endpoint did not start');
}

const ws = new WebSocket(await getWebSocketDebuggerUrl());
let nextId = 1;
const pending = new Map();
const browserLogs = [];

ws.addEventListener('message', (event) => {
  const raw = typeof event.data === 'string' ? event.data : Buffer.from(event.data).toString('utf8');
  const message = JSON.parse(raw);
  if (message.id && pending.has(message.id)) {
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolve(message.result);
  } else if (message.method === 'Runtime.consoleAPICalled') {
    browserLogs.push(message.params.args.map((arg) => arg.value || arg.description || '').join(' '));
  } else if (message.method === 'Runtime.exceptionThrown') {
    browserLogs.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
  }
});

await new Promise((resolve) => ws.addEventListener('open', resolve, { once: true }));

function cdp(method, params = {}) {
  const id = nextId;
  nextId += 1;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`CDP timeout: ${method}`));
    }, 5000);
    pending.set(id, {
      resolve: (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      reject: (error) => {
        clearTimeout(timer);
        reject(error);
      },
    });
  });
}

async function evaluate(expression, awaitPromise = true) {
  const result = await cdp('Runtime.evaluate', {
    expression,
    awaitPromise,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    const detail = result.exceptionDetails.exception?.description || result.exceptionDetails.text || 'Runtime evaluation failed';
    throw new Error(detail);
  }
  return result.result.value;
}

const settingsBase = {
  enabled: true,
  highlightEnabled: true,
  labelsEnabled: true,
  toxicityFilterEnabled: false,
  confidenceThreshold: 0.15,
  sensitivity: 0.5,
  theme: 'system',
  customColors: {},
  enabledEmotions: ['joy', 'anger', 'sadness', 'anxiety', 'fear', 'surprise', 'neutral', 'toxic', 'sarcastic'],
  backendApiUrl: 'http://localhost:8001',
  localOnly: false,
  cacheEnabled: true,
  maxCacheSize: 500,
};

const modes = [
  ['emotion_en', 'Joy', 'Multi-label emotions'],
  ['emotion_vi', 'Joy', 'VI->EN multi-label emotions'],
  ['mental_health_en', 'Depression', 'Mental health screening'],
  ['mental_health_vi', 'Depression', 'VI->EN mental health screening'],
];

const results = [];

async function injectContentWithMode({ mode, text, url, youtube = false }) {
  await cdp('Page.navigate', { url });
  await delay(500);
  await evaluate(`(() => {
    if (!document.body) {
      document.documentElement.appendChild(document.createElement('body'));
    }
    document.body.replaceChildren();
    if (${JSON.stringify(youtube)}) {
      const thread = document.createElement('ytd-comment-thread-renderer');
      const comment = document.createElement('yt-formatted-string');
      comment.id = 'content-text';
      comment.textContent = ${JSON.stringify(text)};
      thread.appendChild(comment);
      document.body.appendChild(thread);
      return;
    }
    const main = document.createElement('main');
    const article = document.createElement('article');
    const paragraph = document.createElement('p');
    paragraph.textContent = ${JSON.stringify(text)};
    article.appendChild(paragraph);
    main.appendChild(article);
    document.body.appendChild(main);
  })()`);
  await delay(250);

  const settings = { ...settingsBase, activeMode: mode };
  const mock = `
    window.__settings = ${JSON.stringify(settings)};
    window.fetch = async () => ({
      ok: true,
      json: async () => ({ translated_text: 'I feel sad and exhausted' })
    });
    window.chrome = {
      storage: {
        sync: {
          get: async () => ({ emotionLensSettings: window.__settings }),
          set: async () => undefined
        },
        onChanged: { addListener: () => undefined }
      },
      runtime: {
        sendMessage: async (message) => {
          if (message.type === 'ANALYZE_EMOTION') {
            return { payload: {
              primary_emotion: 'joy',
              confidence: 0.91,
              label_type: 'fine',
              language: 'en',
              scores_28: { joy: 0.91, neutral: 0.02 },
              scores_9: { joy: 0.91, neutral: 0.02 },
              toxicity_score: 0,
              sarcasm_score: 0,
              num_labels: 28
            } };
          }
          return {};
        },
        onMessage: { addListener: () => undefined }
      }
    };
  `;
  await evaluate(mock);
  await evaluate(`(() => {
    const fakeCurrentScript = { tagName: 'SCRIPT', src: 'chrome-extension://verify/content.js' };
    Object.defineProperty(document, 'currentScript', { configurable: true, get: () => fakeCurrentScript });
  })()`);
  await evaluate(contentScript);
  await evaluate('delete document.currentScript');
  await delay(1300);

  return evaluate(`(() => {
    const badge = document.querySelector('.emotion-lens-badge');
    return badge ? { text: badge.textContent, title: badge.title } : null;
  })()`);
}

try {
  await cdp('Runtime.enable');
  await cdp('Page.enable');

  for (const [mode, expectedText, expectedTitle] of modes) {
    const html = `<html><body><article><p id="post">${mode.includes('vi') ? 'tôi rất buồn và mệt mỏi' : 'I am happy but also depressed today'}</p></article></body></html>`;
    await cdp('Page.navigate', { url: `data:text/html,${encodeURIComponent(html)}` });
    await delay(250);

    const settings = { ...settingsBase, activeMode: mode };
    const mock = `
      window.__settings = ${JSON.stringify(settings)};
      window.fetch = async () => ({
        ok: true,
        json: async () => ({ translated_text: 'I feel sad and exhausted' })
      });
      window.chrome = {
        storage: {
          sync: {
            get: async () => ({ emotionLensSettings: window.__settings }),
            set: async () => undefined
          },
          onChanged: { addListener: () => undefined }
        },
        runtime: {
          sendMessage: async (message) => {
            if (message.type === 'ANALYZE_EMOTION') {
              return { payload: {
                primary_emotion: 'joy',
                confidence: 0.91,
                label_type: 'fine',
                language: window.__settings.activeMode === 'emotion_vi' ? 'vi' : 'en',
                scores_28: { joy: 0.91, neutral: 0.02 },
                scores_9: { joy: 0.91, neutral: 0.02 },
                toxicity_score: 0,
                sarcasm_score: 0,
                num_labels: 28
              } };
            }
            if (message.type === 'ANALYZE_MENTAL_HEALTH') {
              return { payload: {
                primary_condition: 'Depression',
                primary_confidence: 0.88,
                all_scores: { Normal: 0.02, Depression: 0.88 },
                needs_attention: true,
                severity_level: 4,
                severity_label: 'Severe',
                language: window.__settings.activeMode === 'mental_health_vi' ? 'vi' : 'en'
              } };
            }
            return {};
          },
          onMessage: { addListener: () => undefined }
        }
      };
    `;
    await evaluate(mock);
    await evaluate(`(() => {
      const fakeCurrentScript = { tagName: 'SCRIPT', src: 'chrome-extension://verify/content.js' };
      Object.defineProperty(document, 'currentScript', { configurable: true, get: () => fakeCurrentScript });
      const script = document.createElement('script');
      script.textContent = ${JSON.stringify(contentScript)};
      document.documentElement.appendChild(script);
      script.remove();
      delete document.currentScript;
    })()`);
    await delay(1300);
    const badge = await evaluate(`(() => {
      const badge = document.querySelector('.emotion-lens-badge');
      return badge ? { text: badge.textContent, title: badge.title } : null;
    })()`);
    const ok = Boolean(badge?.text?.includes(expectedText) && badge?.title?.includes(expectedTitle));
    results.push({ mode, ok, badge });
  }

  const platformScenarios = [
    ['facebook', 'https://www.facebook.com.invalid/emotion-lens-verify', 'I am happy on Facebook today', false],
    ['x', 'https://x.com.invalid/emotion_lens_verify/status/1', 'I am happy on X today', false],
    ['tiktok', 'https://www.tiktok.com.invalid/@emotionlens/video/1', 'I am happy on TikTok today', false],
    ['threads', 'https://www.threads.net.invalid/@emotionlens/post/1', 'I am happy on Threads today', false],
    ['reddit', 'https://www.reddit.com.invalid/r/emotionlens/comments/1/test/', 'I am happy on Reddit today', false],
    ['youtube', 'https://www.youtube.com.invalid/watch?v=emotionlens', 'I am happy on YouTube today', true],
  ];

  for (const [platform, url, text, youtube] of platformScenarios) {
    const badge = await injectContentWithMode({ mode: 'emotion_en', text, url, youtube });
    const ok = Boolean(badge?.text?.includes('Joy') && badge?.title?.includes('Multi-label emotions'));
    results.push({ platform, mode: 'emotion_en', ok, badge });
  }
} finally {
  ws.close();
  chrome.kill();
}

if (!results.every((result) => result.ok)) {
  console.error(browserLogs.join('\n'));
  console.error(JSON.stringify(results, null, 2));
  process.exit(1);
}

console.log(JSON.stringify(results, null, 2));
