#!/usr/bin/env python3
"""
Gán nhãn cảm xúc cho file JSONL (sau khi crawl + --save).

Cách 1 — CSV (khuyên dùng, dễ label trên Excel/Google Sheets):
  python label_emotions.py export  data/crawls/labeling/youtube_*_for_labeling.jsonl
  # Điền cột emotion, toxicity (0-1), sarcasm (0-1) trong file CSV
  python label_emotions.py import  data/crawls/labeling/youtube_*_for_labeling.csv

Cách 2 — Gán nhãn tương tác trong terminal (ít dòng):
  python label_emotions.py interactive data/crawls/labeling/youtube_*_for_labeling.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Nhãn emotion khớp pipeline train (ai_nlp) + extension (thêm toxic/sarcastic qua field riêng)
EMOTION_LABELS = [
    "joy",        # vui, hài, tích cực
    "anger",      # giận, khó chịu
    "sadness",    # buồn, thất vọng
    "anxiety",    # lo lắng, stress
    "fear",       # sợ, hoảng
    "surprise",   # ngạc nhiên, shock
    "admiration", # ngưỡng mộ, khen
    "love",       # yêu thương, quan tâm
    "neutral",    # trung tính, thông tin
    "sarcastic",  # mỉa mai, đùa châm biếm (có thể ghi thêm sarcasm=1)
    "toxic",      # độc hại, xúc phạm (hoặc toxicity=1)
]

EMOTION_HELP = """
Nhãn emotion (chọn 1):
  joy        — vui, hài, 😂 tích cực
  anger      — giận, cay, gắt
  sadness    — buồn, thất vọng 😢
  anxiety    — lo, stress, hồi hộp
  fear       — sợ, hoảng
  surprise   — ngạc nhiên, shock
  admiration — khen, ngưỡng mộ
  love       — thương, quan tâm
  neutral    — không rõ cảm xúc / thông tin
  sarcastic  — mỉa mai (=))) ironi
  toxic      — xúc phạm, hate

toxicity, sarcasm: số 0.0–1.0 (tùy chọn, để trống = null)
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_export(input_path: Path, csv_path: Path | None) -> None:
    rows = load_jsonl(input_path)
    out = csv_path or input_path.with_suffix(".csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "text",
                "language",
                "source",
                "text_hash",
                "emotion",
                "toxicity",
                "sarcasm",
                "notes",
            ],
        )
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow({
                "row_id": i,
                "text": row.get("text", ""),
                "language": row.get("language", ""),
                "source": row.get("source", ""),
                "text_hash": row.get("text_hash", ""),
                "emotion": row.get("emotion") or "",
                "toxicity": row.get("toxicity") if row.get("toxicity") is not None else "",
                "sarcasm": row.get("sarcasm") if row.get("sarcasm") is not None else "",
                "notes": "",
            })
    print(f"Exported {len(rows)} rows -> {out}")
    print(EMOTION_HELP)


def cmd_import(csv_path: Path, jsonl_path: Path | None) -> None:
    jsonl_in = jsonl_path or csv_path.with_name(
        csv_path.stem.replace("_labeled", "") + ".jsonl"
    )
    if not jsonl_in.exists():
        # Try matching for_labeling.jsonl
        candidate = csv_path.parent / csv_path.name.replace(".csv", "").replace("_labeled", "")
        if candidate.suffix != ".jsonl":
            candidate = Path(str(candidate) + ".jsonl")
        if candidate.exists():
            jsonl_in = candidate
        else:
            raise SystemExit(f"Không tìm thấy file JSONL gốc. Chỉ định: --jsonl path")

    rows = load_jsonl(jsonl_in)
    labeled = 0
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for item in reader:
            rid = int(item["row_id"])
            if rid < 0 or rid >= len(rows):
                continue
            emo = (item.get("emotion") or "").strip().lower()
            if emo:
                if emo not in EMOTION_LABELS:
                    print(f"Warning row {rid}: unknown emotion '{emo}', vẫn lưu.")
                rows[rid]["emotion"] = emo
                labeled += 1
            tox = (item.get("toxicity") or "").strip()
            if tox != "":
                rows[rid]["toxicity"] = float(tox)
            sar = (item.get("sarcasm") or "").strip()
            if sar != "":
                rows[rid]["sarcasm"] = float(sar)
            note = (item.get("notes") or "").strip()
            if note:
                rows[rid].setdefault("labels", {})["annotator_note"] = note

    out = jsonl_in.parent / (jsonl_in.stem + "_labeled.jsonl")
    save_jsonl(out, rows)
    print(f"Imported {labeled} labels -> {out}")


def cmd_interactive(jsonl_path: Path, start: int = 0) -> None:
    rows = load_jsonl(jsonl_path)
    print(EMOTION_HELP)
    print(f"File: {jsonl_path} ({len(rows)} dòng). Enter = bỏ qua, q = lưu & thoát\n")

    for i in range(start, len(rows)):
        row = rows[i]
        if row.get("emotion"):
            continue
        print(f"\n--- [{i+1}/{len(rows)}] ({row.get('language')}) ---")
        print(row.get("text", "")[:300])
        emo = input("emotion? ").strip().lower()
        if emo == "q":
            break
        if emo:
            row["emotion"] = emo
        tox = input("toxicity 0-1 (Enter=bỏ)? ").strip()
        if tox:
            row["toxicity"] = float(tox)
        sar = input("sarcasm 0-1 (Enter=bỏ)? ").strip()
        if sar:
            row["sarcasm"] = float(sar)

    out = jsonl_path.parent / (jsonl_path.stem + "_labeled.jsonl")
    save_jsonl(out, rows)
    print(f"\nSaved -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Label emotions on crawl JSONL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="JSONL -> CSV for Excel/Sheets")
    p_exp.add_argument("input", type=Path)
    p_exp.add_argument("-o", "--output", type=Path, default=None)

    p_imp = sub.add_parser("import", help="CSV -> labeled JSONL")
    p_imp.add_argument("csv", type=Path)
    p_imp.add_argument("--jsonl", type=Path, default=None)

    p_int = sub.add_parser("interactive", help="Label in terminal")
    p_int.add_argument("input", type=Path)
    p_int.add_argument("--start", type=int, default=0)

    args = parser.parse_args()
    if args.cmd == "export":
        cmd_export(args.input, args.output)
    elif args.cmd == "import":
        cmd_import(args.csv, args.jsonl)
    elif args.cmd == "interactive":
        cmd_interactive(args.input, args.start)


if __name__ == "__main__":
    main()
