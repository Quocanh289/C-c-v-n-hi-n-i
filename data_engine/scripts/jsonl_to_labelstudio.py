#!/usr/bin/env python3
"""Convert *_for_labeling.jsonl → Label Studio import JSON."""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="*_for_labeling.jsonl")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    tasks = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append({
                "data": {
                    "text": row.get("text") or row.get("text_raw", ""),
                    "language": row.get("language", ""),
                    "source": row.get("source", ""),
                    "text_hash": row.get("text_hash", ""),
                    "row_id": i,
                },
            })

    out = args.output or args.input.with_name(args.input.stem + "_labelstudio.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(tasks)} tasks -> {out}")
    print("Label Studio: Import -> Upload JSON -> chọn file trên")


if __name__ == "__main__":
    main()
