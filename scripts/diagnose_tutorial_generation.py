"""Run real TutorialVQA generation while preserving raw model output.

Gold answers and reference intervals are deliberately not passed to retrieval or
generation. They are read only to select the fixed acceptance questions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from run_server import build_real_pipeline

DEFAULT_QUESTIONS = ("dev:1168", "dev:501", "dev:1131")

class TracingService:
    def __init__(self, delegate: Any, sink: list[dict[str, Any]]) -> None:
        self.delegate = delegate
        self.sink = sink
        self.max_pixels = delegate.max_pixels

    def infer(self, content: list[dict[str, Any]], **kwargs: Any) -> str:
        raw = self.delegate.infer(content, **kwargs)
        self.sink.append({"raw": raw})
        print("RAW_GENERATION", raw, flush=True)
        return raw

    def unload(self) -> None:
        self.delegate.unload()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=Path("artifacts/tutorialvqa/bounded-v2/segments.ocr.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/tutorialvqa/bounded-v2/indexes"))
    parser.add_argument("--config", type=Path, default=Path("config.tutorialvqa.toml"))
    parser.add_argument("--evaluation", type=Path, default=Path("artifacts/tutorialvqa/evaluation/dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tutorialvqa/diagnostics/raw-generation.jsonl"))
    parser.add_argument("--question-id", action="append", dest="question_ids")
    args = parser.parse_args()
    evaluation = {
        row["question_id"]: row
        for row in (json.loads(line) for line in args.evaluation.read_text().splitlines() if line.strip())
    }
    pipeline = build_real_pipeline(
        segments_path=args.segments,
        index_dir=args.index_dir,
        config_path=args.config,
        device="cuda",
        low_vram=False,
    )
    generator = pipeline._generator
    traces: list[dict[str, Any]] = []
    generator.service = TracingService(generator.service, traces)
    results = []
    for question_id in tuple(args.question_ids or DEFAULT_QUESTIONS):
        row = evaluation[question_id]
        before = len(traces)
        answer = pipeline.ask(row["question"], [row["video_id"]])
        raw = traces[before]["raw"] if len(traces) > before else None
        result = {
            "question_id": question_id,
            "video_id": row["video_id"],
            "question": row["question"],
            "raw": raw,
            "parsed": answer.to_dict(),
        }
        results.append(result)
        print("PARSED_ANSWER", json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))

if __name__ == "__main__":
    main()
