from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_server import build_real_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers for an evaluation JSONL.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", default=Path("artifacts/answer_predictions.jsonl"), type=Path)
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--index-dir", default=Path("artifacts/indexes"), type=Path)
    parser.add_argument("--config", default=Path("config.toml"), type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()

    pipeline = build_real_pipeline(
        segments_path=args.segments,
        index_dir=args.index_dir,
        config_path=args.config,
        device=args.device,
        low_vram=args.low_vram,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.questions.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            result = pipeline.ask(item["question"])
            destination.write(
                json.dumps(
                    {
                        "question_id": item["question_id"],
                        "answer": result.answer,
                        "abstained": result.abstained,
                        "evidence_segment_ids": [
                            evidence.segment.segment_id for evidence in result.evidence
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            destination.flush()


if __name__ == "__main__":
    main()
