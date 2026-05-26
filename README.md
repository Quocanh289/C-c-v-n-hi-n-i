# Emotion Lens

Emotion Lens là hệ thống phân tích cảm xúc và tín hiệu sức khỏe tinh thần từ văn bản trên web. Dự án gồm bốn phần chính:

- **Chrome Extension**: quét nội dung trên các nền tảng mạng xã hội, hiển thị badge cảm xúc, lưu đoạn text người dùng bôi đen.
- **FastAPI Backend**: cung cấp API phân tích cảm xúc, phân tích mental health, dịch văn bản, lưu lịch sử vào PostgreSQL.
- **Next.js Dashboard**: hiển thị lịch sử text đã lưu, phân phối cảm xúc/vấn đề, tín hiệu cần chú ý và khuyến nghị AI.
- **PostgreSQL Database**: lưu kết quả phân tích, feedback, slang terms, lịch sử text người dùng.

> Lưu ý: dashboard và mental-health recommendation chỉ mang tính hỗ trợ theo dõi thông tin, không phải chẩn đoán y khoa.

---

## Mục Lục

- [Tính Năng Chính](#tính-năng-chính)
- [Kiến Trúc Tổng Quan](#kiến-trúc-tổng-quan)
- [Yêu Cầu Môi Trường](#yêu-cầu-môi-trường)
- [Cài Đặt Database PostgreSQL](#cài-đặt-database-postgresql)
- [Chạy Backend FastAPI](#chạy-backend-fastapi)
- [Chạy Frontend Dashboard](#chạy-frontend-dashboard)
- [Build Và Cài Chrome Extension](#build-và-cài-chrome-extension)
- [Quy Trình Sử Dụng](#quy-trình-sử-dụng)
- [API Chính](#api-chính)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Dữ Liệu Và Model](#dữ-liệu-và-model)
- [Thông Số Model Và Đánh Giá](#thông-số-model-và-đánh-giá)
- [Troubleshooting](#troubleshooting)

---

## Tính Năng Chính

### Chrome Extension

- Hỗ trợ Chrome Manifest V3.
- Popup chọn nhanh 4 chế độ:
  - English Emotion
  - Vietnamese Emotion
  - English Mental Health
  - Vietnamese Mental Health
- Hiển thị danh sách 28 cảm xúc theo GoEmotions cho English/Vietnamese emotion mode.
- Side panel để xem label reference và đổi mode.
- Options page để cấu hình:
  - bật/tắt extension,
  - bật/tắt highlight,
  - bật/tắt label,
  - toxicity filter,
  - backend API URL,
  - confidence threshold.
- Context menu **Save to Emotion Lens** khi bôi đen text.
- Nút **View Analytics Dashboard** mở dashboard với `uid` riêng của người dùng.

### Backend

- FastAPI app tại `backend/app/main.py`.
- API phân tích cảm xúc thường.
- API phân tích mental health.
- API dịch văn bản.
- API lưu text từ extension vào PostgreSQL.
- API dashboard tổng hợp dữ liệu theo `uid`.
- CORS cho frontend local và extension.

### Dashboard

- Next.js 14 + Tailwind CSS.
- URL chính:

```text
http://localhost:3000/dashboard?uid=<UUID_CỦA_BẠN>
```

- Hiển thị:
  - tổng số snippets đã lưu,
  - primary issue,
  - average confidence,
  - overall risk,
  - saved snippets,
  - emotion distribution,
  - issue distribution,
  - signals to watch,
  - actionable recommendations.

### Database

PostgreSQL lưu các bảng chính:

- `analysis_results`
- `feedback_data`
- `slang_terms`
- `unknown_terms`
- `model_versions`
- `user_settings`
- `user_saved_texts`

Bảng `user_saved_texts` là bảng dashboard dùng để lấy lịch sử text đã lưu từ extension.

---

## Kiến Trúc Tổng Quan

```text
Chrome Extension
  ├─ content script: quét DOM, hiển thị badge
  ├─ popup: chọn mode, mở dashboard
  ├─ side panel: label reference
  ├─ options: cấu hình extension
  └─ context menu: Save to Emotion Lens
          │
          ▼
FastAPI Backend
  ├─ /api/analyze
  ├─ /api/mental-health/analyze
  ├─ /api/translate
  ├─ /api/save-text
  └─ /api/dashboard
          │
          ▼
PostgreSQL
  └─ user_saved_texts
          ▲
          │
Next.js Dashboard
  └─ /dashboard?uid=<uuid>
```

---

## Yêu Cầu Môi Trường

Khuyến nghị theo dự án hiện tại:

- Windows 64-bit.
- Python 3.13 64-bit cho backend theo yêu cầu dự án.
- Node.js 18+ hoặc mới hơn.
- PostgreSQL 16+ hoặc Docker Desktop.
- Chrome 109+.
- Không bắt buộc dùng virtual environment.

Kiểm tra nhanh:

```powershell
python --version
node --version
npm --version
docker --version
```

Nếu PowerShell chặn `npm.ps1`, dùng:

```powershell
npm.cmd --version
```

---

## Cài Đặt Database PostgreSQL

Database bắt buộc tên là:

```text
emotionlens
```

### Cách 1: PostgreSQL Local

1. Mở pgAdmin hoặc DataGrip.
2. Kết nối bằng user `postgres`.
3. Tạo database:

```sql
CREATE DATABASE emotionlens;
```

4. Mở database `emotionlens`.
5. Chạy toàn bộ file:

```text
database/init.sql
```

Với DataGrip, nên chọn toàn bộ file rồi dùng **Execute as Single Statement** để tránh lỗi ở các block `$$` của trigger/function.

### Cách 2: Docker Compose

Chạy riêng PostgreSQL bằng compose:

```powershell
$env:POSTGRES_PASSWORD="postgres"
docker compose up -d postgres
```

Kiểm tra container:

```powershell
docker ps
docker logs --tail 40 emotion-lens-postgres
```

Thông tin mặc định trong `docker-compose.yml`:

```text
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=emotionlens
POSTGRES_USER=emotionlens
POSTGRES_PASSWORD=postgres
```

Kiểm tra bảng lưu dashboard:

```powershell
docker exec emotion-lens-postgres psql -U emotionlens -d emotionlens -c "select * from user_saved_texts limit 5;"
```

---

## Chạy Backend FastAPI

Vào thư mục backend:

```powershell
cd backend
```

Cài dependency:

```powershell
pip install -r requirements.txt
```

Tạo file `.env` trong `backend/` nếu chưa có:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=emotionlens
POSTGRES_USER=emotionlens
POSTGRES_PASSWORD=postgres
```

Nếu dùng PostgreSQL local với user `postgres`, đổi lại:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<mật_khẩu_postgres_của_bạn>
```

Chạy server:

```powershell
python -m uvicorn app.main:app --reload
```

Backend mặc định chạy tại:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

Health check:

```powershell
Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing
```

---

## Chạy Frontend Dashboard

Vào thư mục frontend:

```powershell
cd frontend
```

Cài dependency:

```powershell
npm.cmd install
npm.cmd install -D tailwindcss@3 postcss autoprefixer
```

File `.env.local` cần trỏ tới backend:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

Chạy dev server:

```powershell
npm.cmd run dev
```

Frontend chạy tại:

```text
http://localhost:3000
```

Dashboard:

```text
http://localhost:3000/dashboard?uid=<UUID_CỦA_BẠN>
```

---

## Build Và Cài Chrome Extension

Vào thư mục extension:

```powershell
cd extension
```

Cài dependency:

```powershell
npm.cmd install
```

Build production:

```powershell
npm.cmd run build
```

Output nằm ở:

```text
extension/dist
```

Cài vào Chrome:

1. Mở Chrome.
2. Truy cập:

```text
chrome://extensions/
```

3. Bật **Developer mode**.
4. Bấm **Load unpacked**.
5. Chọn thư mục:

```text
extension/dist
```

Khi code extension thay đổi, chạy lại `npm.cmd run build`, sau đó bấm reload extension trong `chrome://extensions/`.

---

## Quy Trình Sử Dụng

### Phân tích cảm xúc trực tiếp trên web

1. Mở trang được hỗ trợ như Facebook, YouTube, Reddit, TikTok, Threads, X/Twitter.
2. Extension content script sẽ quét text/comment.
3. Badge cảm xúc hoặc tín hiệu sẽ được hiển thị nếu bật label/highlight.
4. Chọn mode trong popup hoặc side panel:
   - EN Emotion
   - VI Emotion
   - EN Mental
   - VI Mental

### Lưu text vào dashboard

1. Bôi đen một đoạn text trên web.
2. Chuột phải.
3. Chọn **Save to Emotion Lens**.
4. Extension gửi request:

```text
POST http://localhost:8000/api/save-text
```

5. Backend phân tích text và lưu vào `user_saved_texts`.
6. Mở popup extension.
7. Bấm **View Analytics Dashboard**.
8. Dashboard mở:

```text
http://localhost:3000/dashboard?uid=<UUID_CỦA_EXTENSION>
```

---

## API Chính

### Health

```http
GET /api/health
```

### Phân tích cảm xúc

```http
POST /api/analyze
```

Body ví dụ:

```json
{
  "text": "I feel happy today",
  "output_mode": "fine",
  "return_all_probs": true
}
```

### Phân tích mental health

```http
POST /api/mental-health/analyze
```

Body ví dụ:

```json
{
  "text": "I feel anxious and exhausted lately"
}
```

### Dịch văn bản

```http
POST /api/translate
```

### Lưu text từ extension

```http
POST /api/save-text
```

Body:

```json
{
  "uid": "00000000-0000-4000-8000-000000000000",
  "text": "I feel anxious today but I am trying to stay calm.",
  "sourceUrl": "https://example.com"
}
```

Response ví dụ:

```json
{
  "status": "success",
  "message": "Text snippet saved and analyzed successfully",
  "predicted_issue": "Anxiety",
  "confidence": 0.62,
  "emotion_nuances": [
    {
      "key": "anxiety",
      "label": "Anxiety",
      "score": 0.55
    }
  ]
}
```

### Lấy dữ liệu dashboard

```http
GET /api/dashboard?uid=<uuid>
```

Response gồm:

- `saved_texts`
- `aggregated_analysis`
  - `total_snippets`
  - `dominant_issue`
  - `average_confidence`
  - `risk_level`
  - `ai_summary`
  - `emotion_distribution`
  - `issue_distribution`
  - `potential_conditions`
  - `recommendations`
  - `disclaimer`

---

## Cấu Trúc Thư Mục

```text
C-c-v-n-hi-n-i/
├── ai_nlp/
│   └── training/
│       ├── checkpoints/
│       │   ├── emotion_model/
│       │   └── mental_health_model/
│       └── data/
│           └── mental_health/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── models/
│   │   └── routes/
│   │       ├── analyze.py
│   │       ├── dashboard.py
│   │       ├── health.py
│   │       ├── learning.py
│   │       ├── mental_health.py
│   │       ├── slang.py
│   │       └── translate.py
│   └── requirements.txt
├── database/
│   └── init.sql
├── extension/
│   ├── manifest.json
│   ├── webpack.config.js
│   ├── src/
│   │   ├── background/
│   │   ├── content/
│   │   ├── inference/
│   │   ├── options/
│   │   ├── popup/
│   │   ├── sidepanel/
│   │   ├── store/
│   │   └── types/
│   └── dist/
├── frontend/
│   ├── app/
│   │   ├── dashboard/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── package.json
│   └── tailwind.config.js
├── docker-compose.yml
└── README.md
```

---

## Dữ Liệu Và Model

### Emotion model

Checkpoint chính:

```text
ai_nlp/training/checkpoints/emotion_model/best_model/
```

Metrics:

```text
ai_nlp/training/checkpoints/emotion_model/best_model/metrics.json
```

Mode emotion hỗ trợ 28 nhãn GoEmotions:

```text
admiration, amusement, anger, annoyance, approval, caring,
confusion, curiosity, desire, disappointment, disapproval,
disgust, embarrassment, excitement, fear, gratitude, grief,
joy, love, nervousness, optimism, pride, realization, relief,
remorse, sadness, surprise, neutral
```

### Mental health model

Checkpoint chính:

```text
ai_nlp/training/checkpoints/mental_health_model/best_model/
```

Metrics:

```text
ai_nlp/training/checkpoints/mental_health_model/best_model/metrics.json
```

Dataset đang mở trong project:

```text
ai_nlp/training/data/mental_health/mental_health_meta_classifier_rule_dataset.csv
ai_nlp/training/data/mental_health/Sentiment_Mental_health_dataset.csv
```

Các label mental health thường dùng:

```text
Normal, Stress, Anxiety, Personality_disorder, Bipolar, Depression, Suicidal
```

---

## Thông Số Model Và Đánh Giá

Các thông số dưới đây được lấy trực tiếp từ:

```text
ai_nlp/training/checkpoints/emotion_model/best_model/metrics.json
ai_nlp/training/checkpoints/mental_health_model/best_model/metrics.json
```

### 1. Emotion model - GoEmotions 28 labels

Mục tiêu: nhận diện cảm xúc chi tiết theo 28 nhãn GoEmotions, phù hợp cho comment, bài viết ngắn và văn bản mạng xã hội.

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0.5119 |
| Micro F1 | 0.5775 |
| Macro F1 / micro avg | 0.5233 |
| Weighted F1 | 0.5824 |
| Macro Precision | 0.5495 |
| Macro Recall | 0.5307 |
| Micro Precision | 0.5262 |
| Micro Recall | 0.6397 |
| Subset Accuracy | 0.3740 |
| Hamming Loss | 0.0393 |
| MCC | 0.5600 |
| Expected Calibration Error | 0.0359 |

Các lớp có F1 tốt:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| gratitude | 0.9717 | 0.8655 | 0.9156 | 357 |
| amusement | 0.7722 | 0.8053 | 0.7884 | 303 |
| love | 0.7103 | 0.8207 | 0.7616 | 251 |
| remorse | 0.7361 | 0.7794 | 0.7571 | 68 |
| admiration | 0.7313 | 0.7029 | 0.7168 | 488 |
| neutral | 0.5714 | 0.8086 | 0.6696 | 1766 |

Các lớp khó hơn:

| Label | Precision | Recall | F1 | Support | Ghi chú |
|---|---:|---:|---:|---:|---|
| realization | 0.2054 | 0.1811 | 0.1925 | 127 | Dễ nhầm với neutral/surprise. |
| disappointment | 0.2477 | 0.3313 | 0.2835 | 163 | Sắc thái gần sadness/disapproval. |
| relief | 0.2963 | 0.4444 | 0.3556 | 18 | Ít mẫu, khó học ổn định. |
| excitement | 0.3462 | 0.3750 | 0.3600 | 96 | Gần joy/amusement. |
| approval | 0.3220 | 0.4282 | 0.3676 | 397 | Dễ lẫn với admiration/neutral. |
| annoyance | 0.2862 | 0.5149 | 0.3679 | 303 | Gần anger/disapproval. |

Ưu điểm:

- Hỗ trợ đủ 28 nhãn cảm xúc chi tiết, phù hợp UI badge và phân tích nuance.
- Micro recall 0.6397 cho thấy model có xu hướng bắt được nhiều tín hiệu cảm xúc.
- Calibration tốt với ECE 0.0359, confidence tương đối ổn để hiển thị cho người dùng.
- Một số lớp phổ biến như gratitude, amusement, love, admiration có F1 cao.
- Hamming loss thấp 0.0393, phù hợp bài toán multi-label có nhiều nhãn âm.

Nhược điểm:

- Accuracy/subset accuracy chưa cao vì bài toán 28-label multi-label rất khó.
- Các nhãn sắc thái mơ hồ như realization, disappointment, relief, approval còn yếu.
- Lớp ít mẫu như grief, relief, nervousness, pride dễ thiếu ổn định.
- Neutral chiếm tần suất cao nên có thể ảnh hưởng cân bằng dự đoán.
- Tiếng Việt cần bước dịch/chuẩn hóa trước khi đưa vào emotion model, nên chất lượng phụ thuộc thêm vào translation.

Hướng cải thiện:

- Tăng dữ liệu cho các lớp hiếm như grief, relief, nervousness, pride.
- Tối ưu threshold riêng từng label thay vì dùng một ngưỡng chung.
- Thêm dữ liệu mạng xã hội tiếng Việt đã gán nhãn trực tiếp để giảm phụ thuộc dịch.
- Thêm calibration/temperature scaling sau huấn luyện nếu muốn dùng confidence cho quyết định UI quan trọng.

### 2. Mental health model - 7 labels

Mục tiêu: phân loại tín hiệu mental health theo 7 nhãn: `Normal`, `Stress`, `Anxiety`, `Personality_disorder`, `Bipolar`, `Depression`, `Suicidal`.

| Chỉ số | Giá trị |
|---|---:|
| Num samples | 1947 |
| Accuracy | 0.7648 |
| Macro F1 | 0.7186 |
| Weighted F1 | 0.7611 |
| Macro Precision | 0.7199 |
| Macro Recall | 0.7353 |
| Weighted Precision | 0.7695 |
| Weighted Recall | 0.7648 |
| MCC | 0.7182 |
| ROC AUC Macro | 0.9419 |
| Avg Loss | 0.3884 |

Hiệu năng theo từng lớp:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9782 | 0.9803 | 0.9792 | 457 |
| Stress | 0.6736 | 0.8994 | 0.7703 | 179 |
| Suicidal | 0.8301 | 0.7681 | 0.7979 | 496 |
| Personality_disorder | 0.6762 | 0.6961 | 0.6860 | 204 |
| Bipolar | 0.5813 | 0.8266 | 0.6826 | 173 |
| Anxiety | 0.6818 | 0.4870 | 0.5682 | 154 |
| Depression | 0.6178 | 0.4894 | 0.5462 | 284 |

Các lớp khó nhất theo F1:

| Thứ tự | Label | F1 | Support |
|---:|---|---:|---:|
| 1 | Depression | 0.5462 | 284 |
| 2 | Anxiety | 0.5682 | 154 |
| 3 | Bipolar | 0.6826 | 173 |
| 4 | Personality_disorder | 0.6860 | 204 |
| 5 | Stress | 0.7703 | 179 |
| 6 | Suicidal | 0.7979 | 496 |
| 7 | Normal | 0.9792 | 457 |

Các nhầm lẫn lớn:

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

Ưu điểm:

- Accuracy 0.7648 và weighted F1 0.7611 đủ tốt cho dashboard hỗ trợ theo dõi tín hiệu.
- ROC AUC macro 0.9419 cho thấy khả năng tách lớp tổng thể tốt.
- Lớp Normal rất mạnh, giúp hạn chế cảnh báo sai trên nội dung bình thường.
- Lớp Suicidal có F1 0.7979 và precision 0.8301, tương đối tốt cho tín hiệu rủi ro cao.
- Stress có recall 0.8994, bắt được nhiều tín hiệu căng thẳng.

Nhược điểm:

- Depression và Anxiety là hai lớp yếu nhất, F1 lần lượt 0.5462 và 0.5682.
- Depression dễ bị nhầm sang Suicidal hoặc Bipolar.
- Anxiety dễ bị nhầm sang Depression, Bipolar hoặc Personality_disorder.
- Bipolar có recall cao nhưng precision thấp hơn, nghĩa là có thể dự đoán Bipolar hơi rộng.
- Không nên dùng kết quả model như chẩn đoán y khoa; chỉ nên coi là tín hiệu hỗ trợ theo dõi.

Hướng cải thiện:

- Bổ sung dữ liệu Depression/Anxiety chất lượng cao để giảm nhầm lẫn.
- Tách pipeline thành screening nhiều bước: phát hiện risk trước, sau đó phân loại issue.
- Thêm rule hoặc threshold riêng cho các lớp rủi ro cao như Suicidal.
- Đánh giá thêm trên dữ liệu tiếng Việt thật thay vì chỉ phụ thuộc dịch sang tiếng Anh.
- Hiển thị recommendation theo hướng “signals to watch” thay vì kết luận bệnh lý.

### 3. Đánh giá tổng thể hệ thống

Ưu điểm:

- Full-stack hoàn chỉnh: extension, backend, database, dashboard.
- Có khả năng chạy end-to-end: bôi đen text, lưu vào PostgreSQL, xem phân tích trên dashboard.
- UI extension có đủ popup, options và side panel.
- Dashboard tổng hợp lịch sử theo `uid`, phù hợp demo cá nhân hóa.
- Hỗ trợ cả English và Vietnamese mode.
- PostgreSQL schema có sẵn, dễ khởi tạo lại bằng `database/init.sql`.

Nhược điểm:

- Cần chạy đồng thời nhiều thành phần: PostgreSQL, FastAPI, Next.js, extension.
- Vietnamese mode vẫn phụ thuộc bước dịch trước khi phân tích.
- Model emotion 28-label còn yếu ở các cảm xúc sắc thái mơ hồ.
- Mental health recommendation cần được hiểu là hỗ trợ thông tin, không phải tư vấn/chẩn đoán.
- Nếu quên reload extension sau khi build, Chrome vẫn dùng bản cũ.
- Nếu cache `.next` lỗi, frontend có thể cần xóa `.next` và chạy lại dev server.

---

## Troubleshooting

### 1. Dashboard báo backend không phản hồi

Kiểm tra backend:

```powershell
Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing
```

Nếu health OK nhưng dashboard lỗi, kiểm tra PostgreSQL:

```powershell
docker ps
docker logs --tail 40 emotion-lens-postgres
```

Kiểm tra endpoint dashboard:

```powershell
Invoke-WebRequest "http://localhost:8000/api/dashboard?uid=00000000-0000-4000-8000-000000000000" -UseBasicParsing
```

### 2. Save to Emotion Lens không hiện trong dashboard

Kiểm tra extension đã build và reload chưa:

```powershell
cd extension
npm.cmd run build
```

Sau đó vào `chrome://extensions/` và bấm reload extension.

Kiểm tra backend endpoint:

```powershell
$body = @{
  uid = "00000000-0000-4000-8000-000000000000"
  text = "I feel anxious today"
  sourceUrl = "https://example.com"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/save-text" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing
```

Kiểm tra DB:

```powershell
docker exec emotion-lens-postgres psql -U emotionlens -d emotionlens -c "select user_id, left(source_text, 80), predicted_issue, confidence, created_at from user_saved_texts order by created_at desc limit 5;"
```

### 3. Next.js lỗi `Cannot find module './xxx.js'`

Thường do cache `.next` bị lệch khi build/dev server cùng lúc.

Xử lý:

```powershell
cd frontend
```

Dừng process đang giữ port 3000, xóa cache:

```powershell
Remove-Item .next -Recurse -Force
npm.cmd run dev
```

### 4. PowerShell chặn npm

Nếu gặp lỗi `npm.ps1 cannot be loaded`, dùng:

```powershell
npm.cmd install
npm.cmd run build
npm.cmd run dev
```

### 5. Docker không kết nối được

Mở Docker Desktop trước, sau đó chạy:

```powershell
docker info
docker compose up -d postgres
```

### 6. Port đã bị chiếm

Kiểm tra process:

```powershell
netstat -ano | Select-String ":3000"
netstat -ano | Select-String ":8000"
netstat -ano | Select-String ":5432"
```

Dừng process theo PID nếu cần:

```powershell
taskkill /F /T /PID <PID>
```

---

## Lệnh Nhanh Khi Demo

Terminal 1: PostgreSQL bằng Docker

```powershell
$env:POSTGRES_PASSWORD="postgres"
docker compose up -d postgres
```

Terminal 2: Backend

```powershell
cd backend
python -m uvicorn app.main:app --reload
```

Terminal 3: Frontend

```powershell
cd frontend
npm.cmd run dev
```

Terminal 4: Extension build

```powershell
cd extension
npm.cmd run build
```

Sau đó reload extension trong Chrome.

---

## Ghi Chú Phát Triển

- Không commit file `.env` chứa mật khẩu thật.
- Khi đổi code extension phải build lại và reload extension.
- Khi đổi backend nếu chạy không `--reload`, cần restart FastAPI.
- Khi đổi dashboard nếu gặp lỗi cache, xóa `frontend/.next`.
- Recommendation trong dashboard là logic tổng hợp từ lịch sử text đã lưu, không phải tư vấn y tế.

---


