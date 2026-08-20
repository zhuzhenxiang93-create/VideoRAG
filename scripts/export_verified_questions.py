from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from video_rag.evaluation.dataset_validation import read_jsonl, validate_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Export accepted append-only review events as verified JSONL.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=Path("artifacts/annotations/review_events.jsonl"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.supplement_a.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/questions.zh.verified.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation/verified_export.json"))
    args = parser.parse_args()
    candidate_sha = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
    candidates = {item["question_id"]: item for item in read_jsonl(args.candidates)}
    events = read_jsonl(args.events) if args.events.exists() else []
    latest = {
        item["question_id"]: item
        for item in events
        if item.get("candidate_file_sha256") == candidate_sha and item.get("question_id") in candidates
    }
    accepted = [latest[question_id]["annotation_after"] for question_id in candidates if latest.get(question_id, {}).get("action") == "accept"]
    segments = read_jsonl(args.segments)
    report = validate_questions(accepted, segments)
    if not report.valid:
        raise ValueError("verified export failed validation: " + "; ".join(report.errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in accepted)
    args.output.write_text(content, encoding="utf-8")
    summary = {
        **report.to_dict(),
        "candidate_file": str(args.candidates),
        "candidate_file_sha256": candidate_sha,
        "review_event_file": str(args.events),
        "review_event_file_sha256": hashlib.sha256(args.events.read_bytes()).hexdigest() if args.events.exists() else None,
        "accepted": len(accepted),
        "rejected": sum(item.get("action") == "reject" for item in latest.values()),
        "pending": len(candidates) - len(latest),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "formal_metric_eligible": len(accepted) >= 100,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
