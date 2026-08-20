from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from scripts.freeze_verified_splits import SPLITS, assign_video_groups
from video_rag.evaluation.dataset_validation import ALLOWED_TYPES, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run video-disjoint split feasibility without freezing a split.")
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.zh.review_queue.v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/split_feasibility_candidates_v1.json"))
    args = parser.parse_args()
    questions = read_jsonl(args.questions)
    for item in questions:
        item.pop("_line_number", None)
    assignments = assign_video_groups(questions)
    type_videos: defaultdict[str, Counter] = defaultdict(Counter)
    for item in questions:
        type_videos[item["question_type"]][item["video_id"]] += 1
    unanswerable_videos = Counter(item["video_id"] for item in questions if not item["answerable"])
    distribution = {}
    for split in SPLITS:
        rows = [item for item in questions if assignments[item["video_id"]] == split]
        distribution[split] = {
            "questions": len(rows),
            "videos": sorted({item["video_id"] for item in rows}),
            "type_counts": dict(Counter(item["question_type"] for item in rows)),
            "answerable_counts": dict(Counter(str(item["answerable"]).lower() for item in rows)),
        }
    missing = {split: sorted(ALLOWED_TYPES - set(value["type_counts"])) for split, value in distribution.items()}
    per_type = {
        question_type: {
            "questions": sum(counts.values()), "video_count": len(counts),
            "per_video": dict(sorted(counts.items())), "minimum_three_video_condition": len(counts) >= 3,
            "preferred_six_video_condition": len(counts) >= 6,
        }
        for question_type, counts in sorted(type_videos.items())
    }
    report = {
        "status": "candidate_dry_run_only", "formal_split_frozen": False,
        "question_file": str(args.questions), "question_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "question_count": len(questions), "video_count": len({item["video_id"] for item in questions}),
        "per_type": per_type,
        "unanswerable": {
            "questions": sum(unanswerable_videos.values()), "video_count": len(unanswerable_videos),
            "per_video": dict(sorted(unanswerable_videos.items())),
            "minimum_three_video_condition": len(unanswerable_videos) >= 3,
        },
        "dry_run_distribution": distribution, "missing_types_by_split": missing,
        "feasible_on_current_candidate_pool": all(not values for values in missing.values())
            and all(value["minimum_three_video_condition"] for value in per_type.values())
            and len(unanswerable_videos) >= 3,
        "caveat": "candidate feasibility does not guarantee feasibility after human rejection; rerun on verified set before freezing",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
