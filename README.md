# Hệ thống Phân tích Văn bản nhằm Nhận diện Cảm xúc, Biểu hiện Tâm lý và Gợi ý các Vấn đề Sức khỏe Tinh thần Liên quan

Đây là một hệ thống phân tích văn bản tiếng Việt/tiếng Anh để xác định cảm xúc, sắc thái tâm lý, mức độ rủi ro và gợi ý các vấn đề sức khỏe tâm thần liên quan.

## Tổng quan Dự án

### Định nghĩa lại Bài toán
Thay vì: Phân tích cảm xúc và hành vi tâm lý từ text + video realtime

Nên đổi thành: Xây dựng hệ thống phân tích văn bản tiếng Việt/tiếng Anh nhằm nhận diện cảm xúc, sắc thái tâm lý, mức độ rủi ro và gợi ý các vấn đề/bệnh lý tâm thần có thể liên quan đến biểu hiện trong văn bản.

**Lưu ý quan trọng**: Hệ thống không chẩn đoán bệnh, chỉ đưa ra khả năng liên quan / dấu hiệu cần chú ý. WHO và APA đều nhấn mạnh AI trong sức khỏe tâm thần cần được dùng thận trọng, có kiểm chứng, không thay thế chuyên gia lâm sàng.

### Input / Output Mới
**Input**: Người dùng nhập text, ví dụ: "Dạo này tôi không muốn gặp ai, mất ngủ, thấy mình vô dụng và không còn hứng thú với mọi thứ."

**Output**:
```json
{
 "sentiment": "negative",
 "emotions": ["sadness", "hopelessness", "loneliness"],
 "severity": "high",
 "risk_signals": [
   "mất ngủ",
   "tự đánh giá bản thân tiêu cực",
   "mất hứng thú",
   "cô lập xã hội"
 ],
 "possible_related_conditions": [
   {
     "name": "Trầm cảm",
     "confidence": 0.78,
     "reason": "Có dấu hiệu buồn kéo dài, mất hứng thú, cảm giác vô dụng"
   },
   {
     "name": "Rối loạn lo âu",
     "confidence": 0.42,
     "reason": "Có thể liên quan nếu đi kèm căng thẳng, mất ngủ, suy nghĩ tiêu cực"
   }
 ],
 "recommendation": "Đây không phải chẩn đoán y khoa. Nếu tình trạng kéo dài hoặc có ý nghĩ tự làm hại bản thân, nên liên hệ chuyên gia tâm lý hoặc đường dây hỗ trợ khẩn cấp."
}
```

### Luồng Xử lý Hệ thống
1. **Nhận input text**: Người dùng nhập văn bản tự do.
2. **Tiền xử lý văn bản**: Chuẩn hóa, tách từ, phát hiện ngôn ngữ, loại bỏ spam.
3. **Phân tích cảm xúc**: Dự đoán sentiment, emotion, intensity.
4. **Trích xuất biểu hiện tâm lý**: Tìm dấu hiệu như mất ngủ, mất hứng thú, v.v.
5. **Suy luận vấn đề/bệnh lý liên quan**: Ánh xạ biểu hiện sang các vấn đề có thể liên quan.
6. **Tính mức độ rủi ro**: Phân loại thành low/medium/high/critical.
7. **Trả kết quả có giải thích**: Bao gồm cảm xúc, biểu hiện, vấn đề liên quan, khuyến nghị.

### Kiến trúc Đề xuất
Frontend → Backend API → Text Preprocessing → Emotion + Sentiment Model → Symptom Extraction Module → Mental Health Mapping Module → Risk Assessment Module → Explanation Generator → Result JSON + UI Display

### MVP Nên Làm Trước
- Nhập text
- Phân tích sentiment
- Phân tích emotion
- Trích xuất keyword biểu hiện
- Mapping sang vấn đề tâm lý liên quan
- Trả kết quả JSON
- Hiển thị lên giao diện

## Cấu trúc Dự án

- `frontend/`: Giao diện người dùng React/Next.js để nhập văn bản và hiển thị kết quả
- `backend/`: API backend FastAPI để xử lý yêu cầu
- `ai_nlp/`: Các module Python để xử lý NLP, phân tích cảm xúc, phát hiện cảm xúc, v.v.

## Thiết lập Môi trường

### Thiết lập Môi trường với Miniconda
```bash
# Tạo môi trường conda
conda create -n mental_health_analyzer python=3.11 -y

# Kích hoạt môi trường
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mental_health_analyzer

# Cài đặt dependencies cho backend
cd backend
pip install -r requirements.txt

# Cài đặt dependencies cho AI/NLP
cd ../ai_nlp
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
conda run -n mental_health_analyzer uvicorn app.main:app --reload
```

## Sử dụng API

POST /api/analyze

Yêu cầu:
```json
{
  "text": "Tôi thấy mệt mỏi, không muốn nói chuyện với ai..."
}
```

Phản hồi:
```json
{
  "sentiment": "negative",
  "emotions": ["sadness", "hopelessness"],
  "severity": "high",
  "risk_signals": ["mất ngủ", "tự đánh giá bản thân tiêu cực"],
  "possible_related_conditions": [
    {
      "name": "Trầm cảm",
      "confidence": 0.78,
      "reason": "Có dấu hiệu buồn kéo dài, mất hứng thú"
    }
  ],
  "recommendation": "Đây không phải chẩn đoán y khoa..."
}
```

## Phân chia Công việc

- Người 1 (Tú): Frontend + UI hiển thị kết quả (Extension: về nghiên cứu thêm làm sao để có truy cập nhanh)
- Người 2: Backend API + luồng xử lý
- Người 3 (Dũng): AI/NLP Models + quy tắc ánh xạ

## Lưu ý Quan trọng

Hệ thống này không chẩn đoán các vấn đề sức khỏe tâm thần. Nó chỉ cung cấp gợi ý dựa trên phân tích văn bản và nên được sử dụng thận trọng. Luôn tham khảo ý kiến chuyên gia để chẩn đoán thực tế.