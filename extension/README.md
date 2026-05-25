# Emotion Lens Extension Setup

## Requirements

- Google Chrome 109+
- Node.js and npm
- Backend running at `http://localhost:8001` for backend inference modes

## Install Dependencies

From the project root:

```powershell
cd C:\Users\Dung\C-c-v-n-hi-n-i\extension
npm install
```

## Build Extension

```powershell
npm.cmd run build
```

The built Chrome extension is generated in:

```text
C:\Users\Dung\C-c-v-n-hi-n-i\extension\dist
```

## Load In Chrome

1. Open Chrome.
2. Go to `chrome://extensions`.
3. Enable `Developer mode`.
4. Click `Load unpacked`.
5. Select:

```text
C:\Users\Dung\C-c-v-n-hi-n-i\extension\dist
```

After every code change and rebuild, return to `chrome://extensions` and click `Reload` on the extension.

## Run Backend

The extension defaults to backend URL `http://localhost:8001`.

From the project root:

```powershell
cd C:\Users\Dung\C-c-v-n-hi-n-i

$env:PYTHONPATH="C:\Users\Dung\C-c-v-n-hi-n-i\backend"
C:\Users\Dung\anaconda3\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --app-dir C:\Users\Dung\C-c-v-n-hi-n-i\backend
```

Check backend health:

```powershell
curl http://localhost:8001/api/health
```

## Extension Modes

The extension supports 4 active scan modes:

- `EN Emotion`: English multi-label emotion detection
- `VI Emotion`: Vietnamese text translated to English, then emotion detection
- `EN Mental`: English mental-health screening
- `VI Mental`: Vietnamese text translated to English, then mental-health screening

Only one mode scans at a time.

## Configure Backend URL

1. Open the extension popup.
2. Click `Settings`.
3. Set `Backend API URL` to:

```text
http://localhost:8001
```

4. Ensure `Extension Enabled`, `Show Highlights`, and `Show Labels` are enabled.

## Verify Build

From the project root:

```powershell
node scripts\verify_extension_badges.mjs
```

Expected result: all 4 modes and supported platforms return `"ok": true`.

## Supported Platforms

The extension is configured for:

- Facebook
- YouTube
- Reddit
- TikTok
- Threads
- Twitter/X

The content script scans posts and comments, while excluding chat boxes, DMs, and text input areas.

## Troubleshooting

If badges only show on one platform:

1. Rebuild:

```powershell
cd C:\Users\Dung\C-c-v-n-hi-n-i\extension
npm.cmd run build
```

2. Reload the extension in `chrome://extensions`.
3. Refresh the social media page.
4. Check that the selected mode matches the language of the content.

If backend calls fail:

1. Confirm backend is running on port `8001`.
2. Open:

```text
http://localhost:8001/api/health
```

3. Confirm extension settings use `http://localhost:8001`.

