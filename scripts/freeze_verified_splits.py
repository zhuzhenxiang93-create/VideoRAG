from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from video_rag.evaluation.dataset_validation import ALLOWED_TYPES, read_jsonl, validate_questions


SPLITS = ("development", "validation", "test")
RATIOS = {"development": 0.70, "validation": 0.15, "test": 0.15}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_video_groups(questions: list[dict]) -> dict[str, str]:
    by_video: defaultdict[str, list[dict]] = defaultdict(list)
    for item in questions:
        by_video[item["video_id"]].append(item)
    total = len(questions)
    total_types = Counter(item["question_type"] for item in questions)
    split_total = Counter()
    split_types: dict[str, Counter] = {name: Counter() for name in SPLITS}
    assignments = {}

    def cost(split: str, rows: list[dict]) -> float:
        projected_total = split_total[split] + len(rows)
        target_total = max(1.0, total * RATIOS[split])
        value = ((projected_total - target_total) / target_total) ** 2
        row_types = Counter(item["question_type"] for item in rows)
        for question_type, count in total_types.items():
            target = max(1.0, count * RATIOS[split])
            projected = split_types[split][question_type] + row_types[question_type]
            value += ((projected - target) / target) ** 2
        return value

    videos = sorted(by_video, key=lambda value: (-len(by_video[value]), hashlib.sha256(value.encode()).hexdigest()))
    # Seed every split with one video so no split is accidentally empty.
    for index, video_id in enumerate(videos):
        rows = by_video[video_id]
        split = SPLITS[index] if index < len(SPLITS) else min(SPLITS, key=lambda name: (cost(name, rows), SPLITS.index(name)))
        assignments[video_id] = split
        split_total[split] += len(rows)
        split_types[split].update(item["question_type"] for item in rows)
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze deterministic video-disjoint verified splits.")
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.zh.verified.jsonl"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.supplement_a.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/questions.zh.verified.split.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/evaluation/verified_split_manifest.json"))
    parser.add_argument("--minimum", type=int, default=100)
    args = parser.parse_args()
    questions = read_jsonl(args.questions)
    for item in questions:
        item.pop("_line_number", None)
    if len(questions) < args.minimum:
        raise ValueError(f"need at least {args.minimum} verified questions, found {len(questions)}")
    if any(item.get("verification_status") != "verified" for item in questions):
        raise ValueError("split input must contain only verified questions")
    assignments = assign_video_groups(questions)
    output_rows = [{**item, "split": assignments[item["video_id"]]} for item in questions]
    validation = validate_questions(output_rows, read_jsonl(args.segments), require_verified_splits=True)
    if not validation.valid:
        raise ValueError("split validation failed: " + "; ".join(validation.errors))
    missing_types = {
        split: sorted(set(ALLOWED_TYPES) - {item["question_type"] for item in output_rows if item["split"] == split})
        for split in SPLITS
    }
    if any(missing_types.values()):
        raise ValueError(f"each split must contain every question type; missing={missing_types}")
    content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text(encoding="utf-8") != content:
        raise ValueError("frozen split already exists with different content")
    args.output.write_text(content, encoding="utf-8")
    distribution = {
        split: {
            "questions": sum(item["split"] == split for item in output_rows),
            "videos": sorted(video for video, value in assignments.items() if value == split),
            "types": dict(Counter(item["question_type"] for item in output_rows if item["split"] == split)),
        }
        for split in SPLITS
    }
    manifest = {
        "schema_version": 1, "algorithm": "deterministic greedy stratification with video-disjoint groups",
        "ratios": RATIOS, "input_sha256": sha256(args.questions), "output_sha256": sha256(args.output),
        "question_count": len(output_rows), "distribution": distribution,
        "test_policy": "held-out test SHA is frozen and must not be used for method or threshold selection",
        "validation": validation.to_dict(),
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.manifest.exists() and args.manifest.read_text(encoding="utf-8") != text:
        raise ValueError("frozen split manifest already exists with different content")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
