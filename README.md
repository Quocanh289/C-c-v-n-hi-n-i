# Emotion Lens - AI Social Media Emotion Detector

> **Chrome Extension** phát hiện cảm xúc trên mạng xã hội real-time.
> Dùng **GoEmotions 28-label model** (XLM-RoBERTa + LoRA) cho tiếng Anh.
> Hiển thị nhãn cảm xúc trực tiếp trên Facebook, YouTube, Reddit, TikTok, Twitter/X.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Model](https://img.shields.io/badge/model-GoEmotions_28-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Chrome](https://img.shields.io/badge/chrome-109+-green)

---

## 📸 Tính năng chính

| Tính năng | Mô tả |
|-----------|-------|
| **28 emotions** | Phát hiện 28 cảm xúc chi tiết từ GoEmotions (admiration, amusement, anger, joy, love, sadness, surprise, neutral...) |
| **Real-time badges** | Hiển thị icon + tên cảm xúc ngay trên comment/bài viết khi lướt web |
| **Multi-platform** | Hoạt động trên Facebook, YouTube, Reddit, TikTok, Threads, Twitter/X |
| **Dark/Light mode** | Tự động theo theme trình duyệt |
| **Local + Backend** | Chạy local với keyword matching, hoặc kết nối backend để dùng model thật |

---

## 📋 Mục lục

- [Cài đặt nhanh](#-cài-đặt-nhanh)
- [Build Extension](#-build-extension)
- [Chạy Backend (dùng model thật)](#-chạy-backend-dùng-model-thật)
- [Cấu hình Extension](#-cấu-hình-extension)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [28 Emotions](#-28-emotions)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Cài đặt nhanh

### Yêu cầu
- **Node.js** 18+ (cho extension)
- **Python** 3.10+ (cho backend - optional)
- **Chrome** 109+ (cho extension)

### Bước 1: Build Extension

```bash
# Vào thư mục extension
cd extension

# Cài dependencies
npm install

# Build
npm run build
# Hoặc: npx webpack --mode production
```

### Bước 2: Load vào Chrome

1. Mở Chrome → `chrome://extensions/`
2. Bật **Developer mode** (góc trên phải)
3. Click **Load unpacked**
4. Chọn thư mục: `C:\Users\Dung\C-c-v-n-hi-n-i\extension\dist`

### Bước 3: Dùng thử

- Vào **Facebook, YouTube, Reddit** → các comment sẽ có badge cảm xúc 🎯
- Click icon extension trên toolbar → xem thống kê
- Click chuột phải icon → **Options** → tùy chỉnh settings

---

## 🛠 Build Extension

```bash
# === Production build ===
cd extension
npm install
npx webpack --mode production

# === Development (watch mode) ===
npx webpack --mode development --watch

# Output: extension/dist/
#   - background.js     - Service worker
#   - content.js        - Content script (chạy trên web)
#   - popup.js          - Popup UI
#   - options.js        - Settings page
#   - sidepanel.js      - Side panel
#   - chunk.*.js        - Transformers.js library
```

### File cấu hình
| File | Mục đích |
|------|----------|
| `extension/webpack.config.js` | Webpack config (context, entry, output) |
| `extension/manifest.json` | Chrome extension manifest (MV3) |
| `extension/src/types/emotion.ts` | 28 emotion definitions, colors, icons |
| `extension/src/inference/emotionClassifier.ts` | Emotion classifier (rule-based + Transformers.js) |
| `extension/src/content/contentScript.ts` | DOM observer, badge rendering |

---

## 🧠 Chạy Backend (dùng model thật)

Extension có 2 chế độ:
1. **Local mode** (mặc định) - dùng keyword matching, nhanh nhưng không chính xác bằng
2. **Backend mode** - dùng model **XLM-RoBERTa thật** đã train với 28 emotions

### Model đã train nằm ở:
```
ai_nlp/training/checkpoints/emotion_model/best_model/
├── adapter_model.safetensors   # LoRA weights
├── classifier.pt               # 28-label classifier head
├── config.json                 # Model config
├── thresholds.json             # Optimized thresholds
└── tokenizer.json              # Tokenizer
```

### Chạy Backend

```bash
cd C:\Users\Dung\C-c-v-n-hi-n-i

set PYTHONPATH=C:\Users\Dung\C-c-v-n-hi-n-i\backend
C:\Users\Dung\anaconda3\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --app-dir C:\Users\Dung\C-c-v-n-hi-n-i\backend
```

Backend sẽ start và load model (khoảng 5-10 giây):
```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     GoEmotions 28-label model loaded on cuda
```

### Kiểm tra Backend

```bash
curl -X POST "http://localhost:8001/api/analyze" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"I love this amazing video!\"}"
```

Kết quả trả về 28 emotions scores:
```json
{
  "primary_emotion": "love",
  "scores_28": {
    "admiration": 0.42,
    "joy": 0.38,
    "love": 0.39,
    ...
  },
  "num_labels": 28
}
```

---

## ⚙️ Cấu hình Extension

### Dùng Backend (khuyên dùng để có kết quả chính xác nhất)

1. **Chạy backend** (xem hướng dẫn ở trên)
2. Click chuột phải icon **Emotion Lens** → **Options**
3. **Bỏ tick** ô **"Local Only (no backend)"**
4. Đảm bảo URL là: `http://localhost:8001`
5. Click **Save**
6. Refresh extension ở `chrome://extensions/`

### Tùy chỉnh Settings

| Setting | Mô tả |
|---------|-------|
| **Extension Enabled** | Bật/tắt extension |
| **Show Highlights** | Highlight màu nền cho comment có cảm xúc |
| **Show Labels** | Hiển thị badge icon + tên cảm xúc |
| **Toxicity Filter** | Lọc nội dung độc hại |
| **Local Only** | Chỉ dùng local inference (không gọi backend) |
| **Confidence Threshold** | Ngưỡng tin cậy tối thiểu (0-100%) |
| **Sensitivity** | Độ nhạy cảm biến cảm xúc |
| **Emotion Categories** | Bật/tắt từng emotion trong 28 labels |

---

## 📁 Cấu trúc dự án

```
C-c-v-n-hi-n-i/
├── extension/                          # Chrome Extension
│   ├── manifest.json                   # Extension manifest (MV3)
│   ├── webpack.config.js               # Build config
│   ├── package.json                    # Dependencies
│   ├── public/icons/                   # Extension icons
│   ├── src/
│   │   ├── types/emotion.ts            # 28 emotion types + visuals
│   │   ├── inference/emotionClassifier.ts  # AI classifier
│   │   ├── content/contentScript.ts    # DOM observer + badges
│   │   ├── background/background.ts    # Service worker
│   │   ├── popup/popup.tsx             # Popup UI
│   │   ├── options/options.tsx         # Settings page
│   │   └── sidepanel/sidepanel.tsx     # Side panel
│   └── dist/                           # Build output
│
├── backend/                            # Python Backend API
│   ├── app/
│   │   ├── main.py                     # FastAPI app
│   │   ├── models/inference.py         # 28-label model inference
│   │   └── routes/analyze.py           # Analysis endpoints
│   └── requirements.txt                # Python dependencies
│
├── ai_nlp/training/
│   ├── emotion_pipeline/               # Training pipeline
│   │   ├── config.py                   # 28-label config
│   │   ├── model.py                    # GoEmotionsModel
│   │   ├── trainer.py                  # Trainer
│   │   └── pipeline.py                 # Pipeline orchestrator
│   └── checkpoints/emotion_model/
│       └── best_model/                 # ✅ Best trained model
│           ├── adapter_model.safetensors
│           ├── classifier.pt
│           └── thresholds.json
│
└── frontend/                           # Web demo (Next.js)
    └── app/page.tsx                    # Emotion analyzer UI
```

---

## 🎯 28 Emotions

| # | Emotion | Icon | Color | Group | Ví dụ |
|---|---------|------|-------|-------|-------|
| 1 | Admiration | 👏 | Vàng | admiration | "Amazing work!" |
| 2 | Amusement | 😂 | Xanh | joy | "Lol that's hilarious" |
| 3 | Anger | 😡 | Đỏ | anger | "I'm so angry right now" |
| 4 | Annoyance | 😤 | Cam | anger | "This is so annoying" |
| 5 | Approval | 👍 | Xanh | admiration | "I agree with you" |
| 6 | Caring | 💚 | Xanh | love | "Take care of yourself" |
| 7 | Confusion | 😕 | Tím | surprise | "I don't understand" |
| 8 | Curiosity | 🤔 | Tím | surprise | "I wonder why..." |
| 9 | Desire | 😍 | Hồng | admiration | "I want that so bad" |
| 10 | Disappointment | 😞 | Xanh dương | sadness | "That's disappointing" |
| 11 | Disapproval | 👎 | Cam | anger | "I don't agree" |
| 12 | Disgust | 🤢 | Xanh lá | anger | "That's disgusting" |
| 13 | Embarrassment | 😳 | Hồng | sadness | "I'm so embarrassed" |
| 14 | Excitement | 🤩 | Xanh | joy | "I'm so excited!" |
| 15 | Fear | 😨 | Tím | fear | "I'm scared" |
| 16 | Gratitude | 🙏 | Vàng | admiration | "Thank you so much" |
| 17 | Grief | 😭 | Xanh dương | sadness | "I miss them so much" |
| 18 | Joy | 😊 | Xanh | joy | "I'm so happy!" |
| 19 | Love | ❤️ | Đỏ | love | "I love you" |
| 20 | Nervousness | 😬 | Cam | anxiety | "I'm so nervous" |
| 21 | Optimism | 🌟 | Vàng | admiration | "Everything will be fine" |
| 22 | Pride | 🦁 | Xanh | joy | "I'm proud of you" |
| 23 | Realization | 💡 | Tím | surprise | "Oh I get it now" |
| 24 | Relief | 😌 | Xanh | joy | "Thank god it's over" |
| 25 | Remorse | 😔 | Xanh dương | sadness | "I'm sorry" |
| 26 | Sadness | 😢 | Xanh dương | sadness | "I'm so sad" |
| 27 | Surprise | 😲 | Tím | surprise | "Oh my god!" |
| 28 | Neutral | 😐 | Xám | neutral | "It's okay" |

---

## 📡 API Reference

### `POST /api/analyze` - Phân tích cảm xúc

```json
{
  "text": "I love this!",
  "return_all_probs": true,
  "output_mode": "fine"
}
```

**Parameters:**
- `text` (required): Text cần phân tích
- `output_mode`: `"fine"` (28 labels), `"coarse"` (9 labels), `"auto"` (mặc định)

**Response:**
```json
{
  "primary_emotion": "love",
  "confidence": 0.85,
  "label_type": "fine",
  "language": "en",
  "scores_28": { "admiration": 0.42, "love": 0.85, ... },
  "num_labels": 28
}
```

### `GET /api/analyze/labels` - Danh sách labels

```json
{
  "english": {
    "num_labels": 28,
    "labels": ["admiration", "amusement", ...]
  },
  "vietnamese": {
    "num_labels": 9,
    "labels": ["admiration", "anger", ...],
    "note": "Dedicated Vietnamese training coming soon"
  }
}
```

---

## 🔧 Troubleshooting

### Extension không load được
- **Lỗi "Filenames starting with _"**: Đã fix trong webpack.config.js (dùng `chunk.` prefix)
- **Kiểm tra**: `chrome://extensions/` → bật Developer mode → Load unpacked → chọn `extension/dist`

### Badge không hiện trên web
1. Kiểm tra extension đã bật chưa (click icon → toggle ON)
2. Vào Settings → Check "Show Labels" và "Show Highlights"
3. Mở Console (F12) → xem có log `[EmotionLens]` không

### Backend không start được
```bash
# Lỗi "No module named 'app'"
# Fix: Chạy từ thư mục backend với PYTHONPATH
set PYTHONPATH=C:\Users\Dung\C-c-v-n-hi-n-i\backend
C:\Users\Dung\anaconda3\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --app-dir C:\Users\Dung\C-c-v-n-hi-n-i\backend
```

### Model không load được
- Kiểm tra file model tồn tại: `ai_nlp/training/checkpoints/emotion_model/best_model/`
- Phải có: `adapter_model.safetensors`, `classifier.pt`, `config.json`
- Backend log sẽ hiển thị lỗi chi tiết

---

## 📄 License

MIT License

## 👨‍💻 Developer

**Dũng** - AI/NLP Models, Training Pipeline, Extension