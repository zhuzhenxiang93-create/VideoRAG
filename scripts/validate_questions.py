from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_rag.evaluation.dataset_validation import read_jsonl, validate_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VideoRAG question annotations.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-verified-splits", action="store_true")
    args = parser.parse_args()

    report = validate_questions(
        read_jsonl(args.questions), read_jsonl(args.segments),
        require_verified_splits=args.require_verified_splits,
    )
    text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
