# Mental Health Meta Model

Backend co endpoint:

```text
POST /api/mental-health/analyze
```

Model moi ket hop sentiment, emotion va symptom features theo schema cua file
`mental_health_meta_classifier_rule_dataset.csv`, roi tra ve nhan sang loc chi
tiet. Day la ho tro sang loc, khong phai ket luan chan doan y khoa.

## Dataset

Backend tu tim file CSV theo thu tu:

1. `MENTAL_HEALTH_META_DATASET_PATH`
2. `training/mental_health_meta_classifier_rule_dataset.csv`
3. `/data/mental_health_meta_classifier_rule_dataset.csv` khi chay container
4. File cung ten trong OneDrive cua user hien tai

Neu muon chi dinh file khi chay local:

```powershell
$env:MENTAL_HEALTH_META_DATASET_PATH="C:\Users\LENOVO\OneDrive\Tài liệu\Tong hop dai học\mental_health_meta_classifier_rule_dataset.csv"
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Neu da cai `scikit-learn`, backend train `RandomForestClassifier` khi startup.
Neu chua cai, backend van train model nearest-centroid tu CSV bang Python
standard library, do do van co `dataset_loaded: true`.

## Response

```json
{
  "primary_condition": "Depression",
  "diagnosis_code": "moderate_depression_risk",
  "diagnosis_vi": "Có dấu hiệu rối loạn trầm cảm mức vừa",
  "risk_level": "medium",
  "dataset_loaded": true,
  "source": "csv_meta_random_forest",
  "disclaimer": "Ket qua chi ho tro sang loc, khong phai chan doan y khoa chinh thuc."
}
```

`primary_condition` la nhom tuong thich voi UI cu. `diagnosis_code` la output
15 nhan cua dataset, trong do co:

```text
normal -> Không có dấu hiệu rõ ràng
```

Neu text co tin hieu tu hai, backend uu tien:

```text
suicide_high_risk -> Nguy cơ tự hại cao, cần hỗ trợ khẩn cấp
```
