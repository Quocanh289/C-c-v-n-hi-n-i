#!/usr/bin/env python3
"""Re-run cleaning pipeline on a saved raw JSONL file."""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import orjson

from data_engine.core.models import Platform, RawPost
from data_engine.monitoring.logging import configure_logging, get_logger
from data_engine.processing.pipeline import ProcessingPipeline
from data_engine.storage.pipeline_store import CrawlPipelineStore, new_run_id

configure_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Clean raw crawl JSONL for labeling")
    parser.add_argument("input", help="Path to *_raw.jsonl")
    parser.add_argument("--platform", default="youtube", choices=[p.value for p in Platform])
    args = parser.parse_args()

    pipeline = ProcessingPipeline()
    store = CrawlPipelineStore()
    run_id = new_run_id()
    paths = store._paths(args.platform, run_id)

    stats = {"raw": 0, "clean": 0, "spam": 0, "skipped": 0}

    with open(args.input, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            stats["raw"] += 1
            data = orjson.loads(line)
            post = RawPost(
                platform=Platform(data.get("platform", args.platform)),
                text=data["text"],
                author_id=data.get("author_id"),
                post_id=data.get("post_id"),
                parent_id=data.get("parent_id"),
                url=data.get("url"),
                language_hint=data.get("language_hint"),
                metadata=data.get("metadata", {}),
            )
            processed = pipeline.process(post)
            if not processed:
                stats["skipped"] += 1
                continue
            if processed.is_spam:
                stats["spam"] += 1
                continue
            store.append_clean(paths["clean"], processed)
            store.append_clean(paths["labeling"], processed)
            stats["clean"] += 1

    result = store.finalize_run(args.platform, paths, stats)
    print("Done:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
