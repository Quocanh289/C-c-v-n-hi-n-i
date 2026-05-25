# Model And Extension Overview

Tài liệu này mô tả các model đang có trong project, nhãn mà hệ thống hỗ trợ, chỉ số đánh giá hiện có, điểm mạnh/yếu, ưu nhược điểm, và luồng hoạt động của Chrome extension.

## 1. Tổng Quan Hệ Thống

Project gồm 3 phần chính:

- `backend`: FastAPI API cho emotion analysis, mental-health screening, translate, dashboard.
- `extension`: Chrome extension đọc nội dung bài viết/bình luận trên mạng xã hội và hiển thị badge cảm xúc/sàng lọc.
- `ai_nlp`: pipeline training, checkpoint, model config, metrics.

Luồng runtime chính:

```text
Social media page
  -> extension content script quét DOM
  -> bỏ qua chat box / DM / input
  -> trích text từ post/comment
  -> gửi text tới background script
  -> gọi backend API
  -> nhận kết quả emotion hoặc mental health
  -> render badge + secondary chips cạnh nội dung
```

## 2. Emotion Model

### 2.1 Runtime Model Hiện Tại

Backend emotion runtime nằm ở:

```text
backend/app/models/inference.py
```

Hiện tại đây là model rule-based, có output theo shape của GoEmotions. Nó không phải checkpoint transformer được load trực tiếp trong backend. Lý do thể hiện trong code: repository chưa bundle deployable GoEmotions checkpoint, nên backend dùng rule-based scoring ổn định để phục vụ API và feeding mental-health meta features.

API chính:

```text
POST /api/analyze
POST /api/analyze/batch
GET  /api/analyze/labels
```

### 2.2 Emotion Labels

Emotion fine labels theo GoEmotions 28:

```text
admiration, amusement, anger, annoyance, approval,
caring, confusion, curiosity, desire, disappointment,
disapproval, disgust, embarrassment, excitement, fear,
gratitude, grief, joy, love, nervousness,
optimism, pride, realization, relief, remorse,
sadness, surprise, neutral
```

Coarse labels dùng cho grouping/UI:

```text
admiration, anger, anxiety, fear, joy, love, sadness, surprise, neutral
```

Extension hiện hỗ trợ hiển thị:

- `primary_emotion`: nhãn chính.
- `top_emotions`: danh sách emotion phụ có score đủ cao.
- `scores_28`: score theo 28 nhãn.
- `scores_9`: score theo 9 nhóm coarse.

### 2.3 Chỉ Số Model

Không có metrics transformer chính thức cho emotion runtime trong repository hiện tại. Vì backend runtime là rule-based, không nên báo accuracy/F1 như model học máy.

Thông tin có thể khẳng định:

- Source runtime: `rule_emotion_features`.
- Output shape tương thích GoEmotions.
- Có multi-label display qua `top_emotions`.
- Vietnamese text được translate sang English trước khi phân tích emotion.

### 2.4 Emotion Model Làm Tốt Nhất

Model emotion hiện làm tốt nhất với:

- Text ngắn có keyword cảm xúc rõ ràng.
- Các emotion phổ biến như `joy`, `sadness`, `anger`, `fear`, `anxiety`, `love`, `surprise`.
- Use case UI cần highlight nhẹ và badge trực quan, không cần chẩn đoán phức tạp.
- Nội dung post/comment đơn giản trên mạng xã hội.

Ví dụ thuận lợi:

```text
I am happy but also worried
```

Có thể trả primary emotion và `top_emotions` như joy/anxiety-related labels.

### 2.5 Emotion Model Yếu Nhất

Điểm yếu:

- Không hiểu ngữ cảnh sâu như sarcasm phức tạp, mỉa mai nhiều tầng.
- Không xử lý tốt emotion ẩn, gián tiếp, hoặc phụ thuộc context trước đó.
- Rule-based nên dễ miss từ đồng nghĩa không có trong keyword list.
- Một số mapping fine label có thể hơi thô, ví dụ `joy` coarse map vào fine label đầu tiên cùng group như `amusement`.
- Không có calibrated probability đúng nghĩa thống kê.

### 2.6 Ưu Điểm

- Nhanh.
- Dễ debug.
- Ít phụ thuộc GPU.
- Output ổn định.
- Tốt cho extension demo và UX badge real-time.

### 2.7 Nhược Điểm

- Không phải model semantic sâu.
- Khó đạt độ chính xác cao trên text đa dạng.
- Cần mở rộng keyword/rule hoặc thay bằng checkpoint emotion thật nếu muốn production-grade.

## 3. Mental Health Model

### 3.1 Runtime Model Hiện Tại

Mental health API runtime chính nằm ở:

```text
backend/app/routes/mental_health.py
```

Model chính:

```text
microsoft/deberta-v3-base + LoRA adapter
```

Checkpoint candidates:

```text
ai_nlp/training/checkpoints/mental_health_model/best_model
ai_nlp/training/checkpoints/mental_health_model/session_best_20260523_205703
ai_nlp/training/outputs/mental_health_model
```

API chính:

```text
POST /api/mental-health/analyze
POST /api/mental-health/batch
GET  /api/mental-health/labels
```

### 3.2 Mental Health Labels

Model có 7 UI labels:

```text
Normal
Depression
Anxiety
Bipolar
Stress
Suicidal
Personality_disorder
```

Ngoài primary label, API còn trả:

- `risk_signals`: tín hiệu sàng lọc phụ, không phải chẩn đoán.
- `needs_attention`: có cần chú ý không.
- `severity_level`: mức nghiêm trọng dạng số.
- `severity_label`: nhãn severity.
- `source`: nguồn quyết định, ví dụ `trained_model`, `domain_guard_normal`, `signal_rule_correction`, `safety_rule_override`.

### 3.3 Chỉ Số Best Model

Metrics lấy từ:

```text
ai_nlp/training/outputs/mental_health_model/best_model/metrics.json
```

Tổng quan:

| Metric | Value |
|---|---:|
| Accuracy | 0.7648 |
| Macro F1 | 0.7186 |
| Macro Precision | 0.7199 |
| Macro Recall | 0.7353 |
| Weighted F1 | 0.7611 |
| Weighted Precision | 0.7695 |
| Weighted Recall | 0.7648 |
| MCC | 0.7182 |
| ROC AUC Macro | 0.9419 |
| Test samples | 1947 |
| Avg loss | 0.3884 |

Per-class F1:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9782 | 0.9803 | 0.9792 | 457 |
| Depression | 0.6178 | 0.4894 | 0.5462 | 284 |
| Anxiety | 0.6818 | 0.4870 | 0.5682 | 154 |
| Bipolar | 0.5813 | 0.8266 | 0.6826 | 173 |
| Stress | 0.6736 | 0.8994 | 0.7703 | 179 |
| Suicidal | 0.8301 | 0.7681 | 0.7979 | 496 |
| Personality_disorder | 0.6762 | 0.6961 | 0.6860 | 204 |

Hardest classes theo F1:

```text
Depression: 0.5462
Anxiety: 0.5682
Bipolar: 0.6826
Personality_disorder: 0.6860
```

Strongest classes theo F1:

```text
Normal: 0.9792
Suicidal: 0.7979
Stress: 0.7703
```

### 3.4 Confusion / Lỗi Thường Gặp

Các lỗi nhầm lớn nhất trong metrics:

| True label | Predicted label | Count | Percentage |
|---|---|---:|---:|
| Depression | Suicidal | 56 | 19.7% |
| Depression | Bipolar | 46 | 16.2% |
| Suicidal | Stress | 38 | 7.7% |
| Suicidal | Depression | 30 | 6.0% |
| Suicidal | Personality_disorder | 28 | 5.6% |
| Personality_disorder | Depression | 23 | 11.3% |
| Anxiety | Depression | 20 | 13.0% |
| Anxiety | Bipolar | 20 | 13.0% |
| Depression | Anxiety | 18 | 6.3% |
| Anxiety | Personality_disorder | 17 | 11.0% |

Ý nghĩa thực tế:

- Depression, Anxiety, Bipolar dễ bị lẫn vì biểu hiện text có overlap.
- Suicidal có F1 khá tốt nhưng vẫn có nhầm sang Stress/Depression/Personality_disorder.
- Normal rất mạnh trong evaluation, nhưng trong runtime vẫn cần domain guard để tránh câu ngoài miền bị ép vào label bệnh.

### 3.5 Runtime Guards Và Rule Corrections

Sau refactor, mental health endpoint không dùng argmax thô cho mọi câu. Nó có các lớp bảo vệ:

1. `domain_guard_normal`

   Nếu text không có tín hiệu mental-health rõ ràng, trả `Normal`.

   Ví dụ:

   ```text
   Anh làm về ETF đi anh. Em càng đọc càng rối nên chưa hiểu nó dùng để làm gì.
   ```

   Kết quả mong đợi:

   ```text
   Normal, risk_signals=[]
   ```

2. `safety_rule_override`

   Nếu text có tín hiệu self-harm/suicidal rõ ràng, ưu tiên `Suicidal` thay vì để model argmax nhầm.

3. `signal_rule_correction`

   Nếu model trả label không khớp tín hiệu trong text với confidence thấp, API sửa theo tín hiệu match được.

4. `risk_signals`

   Extension hiển thị tín hiệu sàng lọc phụ. Đây là screening signal, không phải diagnosis.

### 3.6 Mental Health Model Làm Tốt Nhất

Model làm tốt nhất với:

- Phân biệt `Normal`.
- Phát hiện nhóm `Suicidal` tương đối tốt.
- Phát hiện stress/burnout rõ keyword như `stressed`, `burnout`, `can't handle it`.
- Text tiếng Anh có biểu hiện rõ ràng.
- Text có symptom cụ thể: mất ngủ, panic, self-harm, hopelessness.

### 3.7 Mental Health Model Yếu Nhất

Điểm yếu:

- Depression và Anxiety có recall thấp hơn các lớp khác.
- Depression dễ bị nhầm sang Suicidal hoặc Bipolar.
- Anxiety dễ bị nhầm sang Depression, Bipolar, Personality_disorder.
- Text ngoài miền nếu không có guard sẽ bị ép vào một trong 7 nhãn.
- Text tiếng Việt phụ thuộc chất lượng dictionary translation.
- Không nên xem output là chẩn đoán y khoa.

### 3.8 Ưu Điểm

- Có checkpoint thật DeBERTa-v3-base + LoRA.
- Có metrics evaluation rõ ràng.
- Có guard tránh false positive ngoài miền.
- Có safety override cho self-harm.
- API trả `risk_signals`, phù hợp UI hơn so với multi-label diagnosis thô.

### 3.9 Nhược Điểm

- Model vẫn là single-label classifier ở lõi.
- Risk signal hiện là hậu xử lý, không phải multi-label model được train riêng.
- Vietnamese translation còn đơn giản.
- Một số class có F1 trung bình, cần thêm data/threshold calibration nếu muốn production.
- Có thể cần HF cache/network khi load base DeBERTa lần đầu.

## 4. Extension Flow

### 4.1 Files Chính

```text
extension/src/content/contentScript.ts
extension/src/background/background.ts
extension/src/inference/emotionClassifier.ts
extension/src/popup/popup.tsx
extension/src/sidepanel/sidepanel.tsx
extension/src/types/emotion.ts
```

### 4.2 Content Script Flow

Content script chạy trên các platform trong manifest:

```text
Facebook
YouTube
Reddit
TikTok
Threads
Twitter/X
```

Flow:

```text
detectPlatform()
  -> injectStyles()
  -> load settings from chrome.storage
  -> init MutationObserver
  -> init IntersectionObserver
  -> performFullScan()
  -> find platform selectors + generic fallback selectors
  -> extract text
  -> queueAnalysis()
  -> processAnalysisQueue()
  -> applyVisualOverlay()
```

### 4.3 Text Extraction

Extension ưu tiên post/comment selectors:

- YouTube comments
- X/Twitter tweet text
- TikTok comment/video description
- Reddit post/comment body
- Facebook post/comment text
- Generic `article`, `role=article`, class chứa `post/comment`

Nếu platform-specific selector miss do DOM thay đổi, extension fallback sang generic scanner.

### 4.4 Chat Box Ignore

Extension loại trừ:

- `contenteditable`
- textbox/input/textarea/select
- chat/dialog/conversation/message drawer
- Messenger/DM surfaces
- YouTube live chat

Mục tiêu: không đọc cảm xúc từ tin nhắn người dùng trong chat box/DM/composer.

### 4.5 Analysis Modes

Extension có 4 mode:

| Mode | Backend/API | Output UI |
|---|---|---|
| EN Emotion | `/api/analyze` | Primary emotion + emotion chips |
| VI Emotion | `/api/analyze` + VI->EN | Primary emotion + emotion chips |
| EN Mental | `/api/mental-health/analyze` | Primary screening + risk signal chips |
| VI Mental | `/api/mental-health/analyze` + VI->EN | Primary screening + risk signal chips |

Chỉ một mode scan tại một thời điểm.

### 4.6 Backend Calls

Background script nhận message từ content script:

```text
ANALYZE_EMOTION
  -> POST {backendApiUrl}/api/analyze

ANALYZE_MENTAL_HEALTH
  -> POST {backendApiUrl}/api/mental-health/analyze
```

Default backend URL:

```text
http://localhost:8001
```

### 4.7 Badge UI

Badge hiện cạnh text được scan:

```text
[icon PrimaryLabel +N] [Secondary 65%] [Another 52%]
```

Trong đó:

- `PrimaryLabel`: nhãn chính.
- `+N`: số nhãn phụ.
- Chips phụ: `top_emotions` hoặc `risk_signals`.
- Tooltip: giải thích model, confidence, signals.

Mental health badge dùng từ "screening" thay vì "diagnosis".

## 5. Điểm Mạnh Của Toàn Hệ Thống

- Có extension hoạt động trực tiếp trên mạng xã hội.
- Hỗ trợ nhiều platform.
- Có backend API tách biệt.
- Có 4 mode rõ ràng.
- Có guard để giảm false positive mental health ngoài miền.
- Có multi-label style display cho emotion.
- Có risk-signal display cho mental health, tránh show nhiều diagnosis gây hiểu nhầm.
- Có script verify extension badges.

## 6. Điểm Yếu / Rủi Ro Kỹ Thuật

- Emotion runtime backend hiện là rule-based, chưa phải transformer checkpoint thật.
- Mental health lõi vẫn là single-label, risk signal là hậu xử lý.
- Translation VI->EN là dictionary-based, có thể làm mất ngữ nghĩa.
- Một số file có text/emoji bị mojibake do encoding trước đó.
- Extension phụ thuộc DOM mạng xã hội, platform thay đổi markup có thể làm selector giảm hiệu quả.
- Transformer chunk trong extension lớn khoảng 752 KiB, build có warning asset size.
- Mental health model cần được trình bày là screening, không phải chẩn đoán.

## 7. Khuyến Nghị Cải Tiến

### Emotion

- Huấn luyện hoặc bundle checkpoint emotion thật.
- Chuyển emotion model sang multi-label thật nếu có dataset phù hợp.
- Calibrate threshold cho từng label.
- Thêm test set riêng cho Vietnamese social media text.

### Mental Health

- Train multi-label symptom/risk-signal model riêng.
- Giữ primary condition làm UI summary, không show multi-label diagnosis thô.
- Cải thiện Vietnamese translation hoặc dùng multilingual model trực tiếp.
- Thêm threshold calibration sau softmax.
- Tách `screening_signals` khỏi `primary_condition` trong UI/analytics.

### Extension

- Thêm telemetry debug local cho số element scanned/filtered.
- Thêm option bật/tắt từng platform.
- Thêm trạng thái "backend offline" trong popup.
- Thêm Playwright/E2E tests cho popup/options/content script.

## 8. Các Lệnh Kiểm Tra Liên Quan

Build extension:

```powershell
cd C:\Users\Dung\C-c-v-n-hi-n-i\extension
npm.cmd run build
```

Typecheck extension:

```powershell
cd C:\Users\Dung\C-c-v-n-hi-n-i\extension
npm.cmd run lint
```

Verify extension badges:

```powershell
cd C:\Users\Dung\C-c-v-n-hi-n-i
node scripts\verify_extension_badges.mjs
```

Run backend:

```powershell
cd C:\Users\Dung\C-c-v-n-hi-n-i
$env:PYTHONPATH="C:\Users\Dung\C-c-v-n-hi-n-i\backend"
C:\Users\Dung\anaconda3\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --app-dir C:\Users\Dung\C-c-v-n-hi-n-i\backend
```

Check backend:

```powershell
curl http://localhost:8001/api/health
```

