from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from video_rag.evaluation.dataset_validation import ALLOWED_TYPES, read_jsonl, validate_questions


SPLITS = ("development", "validation", "test")
RATIOS = {"development": 0.70, "validation": 0.15, "test": 0.15}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_video_groups(questions: list[dict]) -> dict[str, str]:
    by_video: defaultdict[str, list[dict]] = defaultdict(list)
    for item in questions:
        by_video[item["video_id"]].append(item)
    parent = {video_id: video_id for video_id in by_video}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    paraphrase_videos: defaultdict[str, set[str]] = defaultdict(set)
    for item in questions:
        if item.get("paraphrase_group_id"):
            paraphrase_videos[str(item["paraphrase_group_id"])].add(item["video_id"])
    for videos_in_group in paraphrase_videos.values():
        ordered = sorted(videos_in_group)
        for value in ordered[1:]:
            union(ordered[0], value)
    component_videos: defaultdict[str, list[str]] = defaultdict(list)
    for video_id in by_video:
        component_videos[find(video_id)].append(video_id)
    component_rows = {
        root: [item for video_id in videos for item in by_video[video_id]]
        for root, videos in component_videos.items()
    }
    total = len(questions)
    total_types = Counter(item["question_type"] for item in questions)
    components = sorted(component_rows, key=lambda value: (-len(component_rows[value]), hashlib.sha256(value.encode()).hexdigest()))
    rng = random.Random(20260821)

    def objective(candidate: dict[str, str]) -> float:
        split_rows = {
            split: [item for component, assigned in candidate.items() if assigned == split for item in component_rows[component]]
            for split in SPLITS
        }
        value = 0.0
        for split, rows in split_rows.items():
            value += 100.0 * ((len(rows) - total * RATIOS[split]) / total) ** 2
            row_types = Counter(item["question_type"] for item in rows)
            for question_type, count in total_types.items():
                value += 10.0 * ((row_types[question_type] - count * RATIOS[split]) / max(1, count)) ** 2
                if row_types[question_type] == 0:
                    value += 1000.0
        return value

    best, best_score = None, float("inf")
    for _ in range(50_000):
        candidate = {
            component: rng.choices(SPLITS, weights=[RATIOS[value] for value in SPLITS], k=1)[0]
            for component in components
        }
        score = objective(candidate)
        if score < best_score:
            best, best_score = candidate, score
    assert best is not None
    return {
        video_id: best[component]
        for component, videos in component_videos.items()
        for video_id in videos
    }


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
