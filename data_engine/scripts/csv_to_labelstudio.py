#!/usr/bin/env python3
"""
CSV labeling file → Label Studio import JSON.
Label Studio thường KHÔNG hiển thị text nếu upload CSV mà không map cột — dùng file JSON này.

Usage:
  python csv_to_labelstudio.py data/crawls/labeling/youtube_20260520_152312_for_labeling.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    tasks = []
    with open(args.csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            tasks.append({
                "data": {
                    "text": text,
                    "language": row.get("language", ""),
                    "source": row.get("source", ""),
                    "text_hash": row.get("text_hash", ""),
                    "row_id": row.get("row_id", str(i)),
                },
            })

    out = args.output or args.csv_path.with_name(
        args.csv_path.stem + "_labelstudio_import.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(tasks)} tasks -> {out}")
    print()
    print("Trong Label Studio:")
    print("  1. Project -> Import")
    print("  2. Upload file JSON (KHONG upload CSV)")
    print("  3. Confirm import -> vao Label All de thay text")


if __name__ == "__main__":
    main()
