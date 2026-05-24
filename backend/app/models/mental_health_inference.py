"""CSV-trained meta-classifier for mental-health screening output.

The supplied dataset contains sentiment, emotion and symptom feature columns
rather than raw posts. This model extracts matching features from input text,
combines them with emotion inference, and trains a RandomForest on the CSV
labels. Results describe screening signals, not a confirmed medical diagnosis.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from app.models.inference import get_inference

logger = logging.getLogger(__name__)

DATASET_FILENAME = "mental_health_meta_classifier_rule_dataset.csv"
DISCLAIMER = "Ket qua chi ho tro sang loc, khong phai chan doan y khoa chinh thuc."

FEATURE_COLUMNS = [
    "sentiment_negative_prob", "sentiment_neutral_prob", "sentiment_positive_prob",
    "emotion_sadness_prob", "emotion_worry_prob", "emotion_fear_prob",
    "emotion_stress_prob", "emotion_anger_prob", "emotion_hopelessness_prob",
    "emotion_panic_prob", "emotion_calm_prob", "symptom_overthinking_score",
    "symptom_sleep_problem_score", "symptom_loss_interest_score",
    "symptom_low_energy_score", "symptom_low_self_worth_score",
    "symptom_restlessness_score", "symptom_avoidance_score",
    "symptom_panic_body_score", "symptom_intrusive_thought_score",
    "symptom_compulsion_score", "symptom_trauma_recall_score",
    "symptom_attention_problem_score", "symptom_mood_swing_score",
    "symptom_social_fear_score", "symptom_eating_concern_score",
    "symptom_self_harm_score", "duration_days", "frequency_score",
    "functional_impairment_score", "final_risk_score",
]

DETAILED_LABELS = [
    "normal", "mild_depression_risk", "moderate_depression_risk",
    "mild_anxiety_risk", "moderate_anxiety_risk", "panic_related_risk",
    "stress_burnout_risk", "ptsd_related_risk", "ocd_related_risk",
    "social_anxiety_risk", "sleep_problem_risk", "bipolar_related_risk",
    "adhd_related_risk", "eating_concern_risk", "suicide_high_risk",
]

LABEL_VI = {
    "normal": "Không có dấu hiệu rõ ràng",
    "mild_depression_risk": "Có dấu hiệu rối loạn trầm cảm mức nhẹ",
    "moderate_depression_risk": "Có dấu hiệu rối loạn trầm cảm mức vừa",
    "mild_anxiety_risk": "Có dấu hiệu rối loạn lo âu mức nhẹ",
    "moderate_anxiety_risk": "Có dấu hiệu rối loạn lo âu mức vừa",
    "panic_related_risk": "Có dấu hiệu liên quan cơn hoảng sợ",
    "stress_burnout_risk": "Có dấu hiệu căng thẳng/burnout",
    "ptsd_related_risk": "Có dấu hiệu liên quan sang chấn/PTSD",
    "ocd_related_risk": "Có dấu hiệu liên quan ám ảnh cưỡng chế",
    "social_anxiety_risk": "Có dấu hiệu lo âu xã hội",
    "sleep_problem_risk": "Có dấu hiệu rối loạn giấc ngủ",
    "bipolar_related_risk": "Có dấu hiệu dao động khí sắc/lưỡng cực",
    "adhd_related_risk": "Có dấu hiệu liên quan khó chú ý/ADHD",
    "eating_concern_risk": "Có dấu hiệu vấn đề ăn uống/hình ảnh cơ thể",
    "suicide_high_risk": "Nguy cơ tự hại cao, cần hỗ trợ khẩn cấp",
}

RISK_LEVELS = {
    "normal": "none",
    "mild_depression_risk": "low", "moderate_depression_risk": "medium",
    "mild_anxiety_risk": "low", "moderate_anxiety_risk": "medium",
    "panic_related_risk": "medium", "stress_burnout_risk": "medium",
    "ptsd_related_risk": "medium", "ocd_related_risk": "medium",
    "social_anxiety_risk": "low", "sleep_problem_risk": "low",
    "bipolar_related_risk": "medium", "adhd_related_risk": "low",
    "eating_concern_risk": "medium", "suicide_high_risk": "high",
}

# Kept for compatibility with the current UI while detailed labels are returned separately.
MENTAL_HEALTH_LABELS = [
    "Normal", "Depression", "Anxiety", "Bipolar", "Stress",
    "Suicidal", "Personality_disorder",
]

COARSE_GROUP = {
    "normal": "Normal",
    "mild_depression_risk": "Depression", "moderate_depression_risk": "Depression",
    "mild_anxiety_risk": "Anxiety", "moderate_anxiety_risk": "Anxiety",
    "panic_related_risk": "Anxiety", "social_anxiety_risk": "Anxiety",
    "stress_burnout_risk": "Stress", "sleep_problem_risk": "Stress",
    "ptsd_related_risk": "Stress", "ocd_related_risk": "Anxiety",
    "bipolar_related_risk": "Bipolar", "adhd_related_risk": "Stress",
    "eating_concern_risk": "Stress", "suicide_high_risk": "Suicidal",
}

SEVERITY = {"none": (0, "healthy"), "low": (1, "low"), "medium": (3, "medium"), "high": (5, "critical")}


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _phrase_score(text: str, phrases: Dict[str, float]) -> float:
    return min(0.99, sum(weight for phrase, weight in phrases.items() if phrase in text))


class MentalHealthInference:
    """Train-on-load meta model backed by sentiment/emotion/rule features."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.model: Any = None
        self.is_loaded = False
        self.loaded_dataset_path: Optional[str] = None
        self.validation_accuracy: Optional[float] = None
        self.model_type = "meta_rule_fallback"
        self.load_model()

    def _dataset_candidates(self) -> Iterable[Path]:
        if self.dataset_path:
            yield self.dataset_path
        configured = os.getenv("MENTAL_HEALTH_META_DATASET_PATH")
        if configured:
            yield Path(configured)
        project_root = Path(__file__).resolve().parents[3]
        yield project_root / "training" / DATASET_FILENAME
        yield Path("/data") / DATASET_FILENAME
        yield from (Path.home() / "OneDrive").rglob(DATASET_FILENAME)

    def load_model(self) -> bool:
        source = next((candidate for candidate in self._dataset_candidates() if candidate.exists()), None)
        if source is None:
            logger.warning("Mental-health CSV not found; deterministic rules will be used.")
            return False

        try:
            train_x, train_y, valid_x, valid_y = [], [], [], []
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    vector = [float(row.get(column, 0.0) or 0.0) for column in FEATURE_COLUMNS]
                    if row.get("split") == "train":
                        train_x.append(vector)
                        train_y.append(row["label"])
                    else:
                        valid_x.append(vector)
                        valid_y.append(row["label"])
            if not train_x:
                raise ValueError("Dataset does not contain training rows.")

            try:
                from sklearn.ensemble import RandomForestClassifier

                self.model = RandomForestClassifier(
                    n_estimators=240,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                self.model.fit(train_x, train_y)
                self.model_type = "csv_meta_random_forest"
            except ImportError:
                self.model = _NearestCentroidModel().fit(train_x, train_y)
                self.model_type = "csv_meta_centroid"
            if valid_x:
                predicted = self.model.predict(valid_x)
                correct = sum(1 for expected, actual in zip(valid_y, predicted) if expected == actual)
                self.validation_accuracy = correct / len(valid_y)
            self.loaded_dataset_path = str(source)
            self.is_loaded = True
            logger.info(
                "Mental-health meta model trained from %s using %s (%s labels, validation accuracy=%s)",
                source,
                self.model_type,
                len(self.model.classes_),
                f"{self.validation_accuracy:.3f}" if self.validation_accuracy is not None else "n/a",
            )
            return True
        except Exception as exc:
            logger.warning("Failed to train mental-health meta model from CSV: %s. Using rules.", exc)
            self.model = None
            self.is_loaded = False
            return False

    def _duration_days(self, text: str) -> int:
        factors = {
            "ngay": 1, "tuan": 7, "thang": 30, "nam": 365,
            "day": 1, "days": 1, "week": 7, "weeks": 7,
            "month": 30, "months": 30, "year": 365, "years": 365,
        }
        found = re.search(r"\b(\d+)\s*(ngay|tuan|thang|nam|days?|weeks?|months?|years?)\b", text)
        if found:
            return int(found.group(1)) * factors[found.group(2)]
        words = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "sau": 6, "bay": 7}
        found = re.search(r"\b(mot|hai|ba|bon|sau|bay)\s+(ngay|tuan|thang|nam)\b", text)
        return words[found.group(1)] * factors[found.group(2)] if found else 1

    def extract_features(self, text: str) -> Dict[str, float]:
        value = _normalize(text)
        emotion = get_inference().classify(text, output_mode="coarse").get("scores_9", {})
        symptoms = {
            "symptom_overthinking_score": _phrase_score(value, {"suy nghi qua nhieu": 0.9, "overthink": 0.9, "cannot stop worrying": 0.8}),
            "symptom_sleep_problem_score": _phrase_score(value, {"mat ngu": 0.9, "khong ngu": 0.8, "ngu qua nhieu": 0.8, "insomnia": 0.9, "cannot sleep": 0.8}),
            "symptom_loss_interest_score": _phrase_score(value, {"mat hung thu": 0.95, "khong con hung thu": 0.95, "lost interest": 0.95, "no longer enjoy": 0.9}),
            "symptom_low_energy_score": _phrase_score(value, {"met moi": 0.75, "kiet suc": 0.9, "khong co nang luong": 0.9, "exhausted": 0.85, "no energy": 0.9}),
            "symptom_low_self_worth_score": _phrase_score(value, {"vo dung": 0.95, "vo gia tri": 0.95, "worthless": 0.95, "guilty": 0.7}),
            "symptom_restlessness_score": _phrase_score(value, {"bon chon": 0.8, "dung ngoi khong yen": 0.85, "restless": 0.85}),
            "symptom_avoidance_score": _phrase_score(value, {"ne tranh": 0.8, "tranh gap": 0.8, "avoid": 0.8, "khong muon gap ai": 0.8}),
            "symptom_panic_body_score": _phrase_score(value, {"hoang loan": 0.9, "tim dap nhanh": 0.8, "kho tho": 0.75, "panic": 0.9, "heart racing": 0.8, "cannot breathe": 0.7}),
            "symptom_intrusive_thought_score": _phrase_score(value, {"y nghi am anh": 0.9, "intrusive thought": 0.9, "obsessive thought": 0.85}),
            "symptom_compulsion_score": _phrase_score(value, {"cuong che": 0.9, "kiem tra lien tuc": 0.85, "rua tay lien tuc": 0.9, "compulsion": 0.9, "repeatedly check": 0.85}),
            "symptom_trauma_recall_score": _phrase_score(value, {"sang chan": 0.9, "hoi tuong": 0.75, "flashback": 0.95, "trauma": 0.85, "nightmare": 0.6}),
            "symptom_attention_problem_score": _phrase_score(value, {"kho tap trung": 0.85, "mat tap trung": 0.9, "cannot focus": 0.9, "distracted": 0.7}),
            "symptom_mood_swing_score": _phrase_score(value, {"dao dong khi sac": 0.95, "luc vui luc buon": 0.85, "ngu rat it ma khong met": 0.95, "mood swing": 0.95, "reckless spending": 0.85}),
            "symptom_social_fear_score": _phrase_score(value, {"so giao tiep": 0.9, "so bi danh gia": 0.9, "social anxiety": 0.9, "fear being judged": 0.9}),
            "symptom_eating_concern_score": _phrase_score(value, {"so tang can": 0.9, "nhin an": 0.85, "body image": 0.85, "fear gaining weight": 0.9}),
            "symptom_self_harm_score": _phrase_score(value, {"tu lam hai": 0.99, "tu sat": 0.99, "muon chet": 0.99, "khong muon song": 0.95, "self harm": 0.99, "suicide": 0.99, "want to die": 0.99, "unsafe with myself": 0.9}),
        }
        negative = _phrase_score(value, {"buon": 0.4, "tuyet vong": 0.6, "tram cam": 0.55, "vo dung": 0.4, "lo lang": 0.3, "sad": 0.4, "depressed": 0.55, "hopeless": 0.6, "worthless": 0.4, "anxious": 0.3})
        positive = _phrase_score(value, {"vui": 0.6, "hanh phuc": 0.7, "binh yen": 0.55, "happy": 0.65, "calm": 0.55, "fine": 0.35})
        neutral = max(0.02, 1.0 - max(negative, positive))
        total = max(0.01, negative + positive + neutral)
        hopelessness = _phrase_score(value, {"tuyet vong": 0.9, "vo vong": 0.9, "hopeless": 0.9, "no hope": 0.9})
        panic = symptoms["symptom_panic_body_score"]
        stress = _phrase_score(value, {"stress": 0.8, "cang thang": 0.75, "qua tai": 0.7, "burnout": 0.95, "ap luc": 0.65})
        duration = self._duration_days(value)
        frequency = _phrase_score(value, {"moi ngay": 0.85, "lien tuc": 0.75, "gan nhu ca ngay": 0.9, "every day": 0.85, "constantly": 0.75})
        impairment = _phrase_score(value, {"khong the lam viec": 0.95, "nghi hoc": 0.85, "khong hoan thanh": 0.8, "cannot work": 0.95, "stopped attending": 0.85, "cannot function": 0.95})
        final_risk = min(0.99, symptoms["symptom_self_harm_score"] * 0.7 + impairment * 0.13 + hopelessness * 0.1 + negative * 0.07)
        features = {
            "sentiment_negative_prob": negative / total,
            "sentiment_neutral_prob": neutral / total,
            "sentiment_positive_prob": positive / total,
            "emotion_sadness_prob": emotion.get("sadness", 0.0),
            "emotion_worry_prob": emotion.get("anxiety", 0.0),
            "emotion_fear_prob": emotion.get("fear", 0.0),
            "emotion_stress_prob": stress,
            "emotion_anger_prob": emotion.get("anger", 0.0),
            "emotion_hopelessness_prob": hopelessness,
            "emotion_panic_prob": panic,
            "emotion_calm_prob": max(emotion.get("joy", 0.0), positive),
            **symptoms,
            "duration_days": float(duration),
            "frequency_score": frequency,
            "functional_impairment_score": impairment,
            "final_risk_score": final_risk,
        }
        return {name: round(float(features[name]), 4) for name in FEATURE_COLUMNS}

    def _rule_prediction(self, features: Dict[str, float]) -> str:
        if features["symptom_self_harm_score"] >= 0.7:
            return "suicide_high_risk"
        if features["symptom_mood_swing_score"] >= 0.65:
            return "bipolar_related_risk"
        if features["symptom_panic_body_score"] >= 0.65:
            return "panic_related_risk"
        if features["symptom_trauma_recall_score"] >= 0.65:
            return "ptsd_related_risk"
        if features["symptom_compulsion_score"] >= 0.65:
            return "ocd_related_risk"
        if features["symptom_social_fear_score"] >= 0.65:
            return "social_anxiety_risk"
        if features["symptom_eating_concern_score"] >= 0.65:
            return "eating_concern_risk"
        if max(features["symptom_loss_interest_score"], features["emotion_sadness_prob"]) >= 0.6:
            return "moderate_depression_risk" if features["duration_days"] >= 14 else "mild_depression_risk"
        if max(features["emotion_worry_prob"], features["symptom_overthinking_score"]) >= 0.6:
            return "moderate_anxiety_risk" if features["duration_days"] >= 14 else "mild_anxiety_risk"
        if features["symptom_sleep_problem_score"] >= 0.65:
            return "sleep_problem_risk"
        if features["emotion_stress_prob"] >= 0.6:
            return "stress_burnout_risk"
        if features["symptom_attention_problem_score"] >= 0.7:
            return "adhd_related_risk"
        return "normal"

    def classify(self, text: str) -> Dict[str, Any]:
        features = self.extract_features(text)
        # Explicit guards make safety and healthy inputs reliable even with synthetic training data.
        active_scores = [features[key] for key in FEATURE_COLUMNS if key.startswith("symptom_")]
        clearly_normal = (
            max(active_scores, default=0.0) < 0.3
            and features["sentiment_negative_prob"] < 0.25
            and features["emotion_stress_prob"] < 0.3
        )
        if features["symptom_self_harm_score"] >= 0.7:
            diagnosis = "suicide_high_risk"
            detailed_scores = {label: 0.0 for label in DETAILED_LABELS}
            detailed_scores[diagnosis] = 1.0
        elif clearly_normal:
            diagnosis = "normal"
            detailed_scores = {label: 0.0 for label in DETAILED_LABELS}
            detailed_scores[diagnosis] = 1.0
        elif self.is_loaded and self.model is not None:
            probabilities = self.model.predict_proba([[features[column] for column in FEATURE_COLUMNS]])[0]
            detailed_scores = {label: 0.0 for label in DETAILED_LABELS}
            for label, probability in zip(self.model.classes_, probabilities):
                detailed_scores[label] = round(float(probability), 4)
            diagnosis = max(detailed_scores, key=detailed_scores.get)
        else:
            diagnosis = self._rule_prediction(features)
            detailed_scores = {label: 0.0 for label in DETAILED_LABELS}
            detailed_scores[diagnosis] = 0.75

        primary_group = COARSE_GROUP[diagnosis]
        grouped_scores = {label: 0.0 for label in MENTAL_HEALTH_LABELS}
        for label, score in detailed_scores.items():
            grouped_scores[COARSE_GROUP[label]] += score
        grouped_scores = {key: round(min(value, 1.0), 4) for key, value in grouped_scores.items()}
        confidence = round(float(detailed_scores[diagnosis]), 4)
        risk = RISK_LEVELS[diagnosis]
        severity_level, severity_label = SEVERITY[risk]
        top = sorted(detailed_scores.items(), key=lambda item: item[1], reverse=True)[:3]
        important_features = {
            key: value for key, value in features.items()
            if value >= 0.5 or key in {"duration_days", "final_risk_score"}
        }
        return {
            "primary_condition": primary_group,
            "primary_confidence": confidence,
            "all_scores": grouped_scores,
            "diagnosis_code": diagnosis,
            "diagnosis_vi": LABEL_VI[diagnosis],
            "diagnosis_scores": detailed_scores,
            "risk_level": risk,
            "needs_attention": diagnosis != "normal",
            "severity_level": severity_level,
            "severity_label": severity_label,
            "top_predictions": [{"label": label, "confidence": score} for label, score in top],
            "extracted_features": important_features,
            "source": self.model_type if self.is_loaded else "meta_rule_fallback",
            "num_labels": len(DETAILED_LABELS),
            "disclaimer": DISCLAIMER,
            "dataset_loaded": self.is_loaded,
            "validation_accuracy": self.validation_accuracy,
        }


class _NearestCentroidModel:
    """Dependency-free trained model used when scikit-learn is unavailable."""

    def __init__(self):
        self.classes_: list[str] = []
        self.centroids: Dict[str, list[float]] = {}
        self.scales: list[float] = []

    def fit(self, samples: list[list[float]], labels: list[str]) -> "_NearestCentroidModel":
        self.classes_ = sorted(set(labels))
        columns = list(zip(*samples))
        self.scales = []
        for values in columns:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            self.scales.append(max(math.sqrt(variance), 0.05))
        for label in self.classes_:
            rows = [sample for sample, target in zip(samples, labels) if target == label]
            self.centroids[label] = [
                sum(row[index] for row in rows) / len(rows)
                for index in range(len(FEATURE_COLUMNS))
            ]
        return self

    def predict_proba(self, samples: list[list[float]]) -> list[list[float]]:
        output = []
        for sample in samples:
            similarities = []
            for label in self.classes_:
                centroid = self.centroids[label]
                distance = math.sqrt(
                    sum(((value - centroid[index]) / self.scales[index]) ** 2 for index, value in enumerate(sample))
                    / len(sample)
                )
                similarities.append(math.exp(-distance * 2.0))
            total = sum(similarities) or 1.0
            output.append([score / total for score in similarities])
        return output

    def predict(self, samples: list[list[float]]) -> list[str]:
        return [
            self.classes_[max(range(len(scores)), key=scores.__getitem__)]
            for scores in self.predict_proba(samples)
        ]


_instance: Optional[MentalHealthInference] = None


def get_mental_health_inference() -> MentalHealthInference:
    global _instance
    if _instance is None:
        _instance = MentalHealthInference()
    return _instance
