from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def perceptual_hash(path: Path) -> int:
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read frame {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    values = cv2.dct(np.float32(resized))[:8, :8]
    threshold = float(np.median(values[1:, :]))
    bits = values > threshold
    result = 0
    for bit in bits.flatten():
        result = (result << 1) | int(bit)
    return result


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit keyframe coverage and duplication.")
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/evaluation/keyframe_audit.json"), type=Path)
    args = parser.parse_args()

    segments = [
        json.loads(line)
        for line in args.segments.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root = Path.cwd()
    unique_frames: dict[str, dict] = {}
    memberships: Counter[str] = Counter()
    per_video_segments: defaultdict[str, list] = defaultdict(list)
    empty_searchable = []
    for segment in segments:
        per_video_segments[segment["video_id"]].append(segment)
        if not (segment.get("transcript", "").strip() or segment.get("visual_caption", "").strip()):
            empty_searchable.append(segment["segment_id"])
        for frame in segment.get("keyframes", []):
            resolved = Path(frame["path"])
            if not resolved.is_absolute():
                resolved = root / resolved
            key = str(resolved.resolve())
            unique_frames.setdefault(
                key,
                {
                    "path": frame["path"],
                    "timestamp": frame["timestamp"],
                    "video_id": segment["video_id"],
                    "exists": resolved.is_file(),
                },
            )
            memberships[key] += 1

    exact_hashes: defaultdict[str, list[str]] = defaultdict(list)
    phashes: dict[str, int] = {}
    unreadable = []
    for key, frame in unique_frames.items():
        path = Path(key)
        if not path.is_file():
            unreadable.append(frame["path"])
            continue
        exact_hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(key)
        try:
            phashes[key] = perceptual_hash(path)
        except ValueError:
            unreadable.append(frame["path"])

    adjacent_distances = []
    per_video_frames: defaultdict[str, list[tuple[float, str]]] = defaultdict(list)
    for key, frame in unique_frames.items():
        if key in phashes:
            per_video_frames[frame["video_id"]].append((frame["timestamp"], key))
    near_duplicate_pairs = []
    for video_id, values in per_video_frames.items():
        values.sort()
        for (left_time, left), (right_time, right) in zip(values, values[1:]):
            distance = hamming(phashes[left], phashes[right])
            adjacent_distances.append(distance)
            if distance <= 6:
                near_duplicate_pairs.append(
                    {"video_id": video_id, "left": left_time, "right": right_time, "hamming": distance}
                )

    frame_counts = [len(item.get("keyframes", [])) for item in segments]
    zero_segments = [item["segment_id"] for item in segments if not item.get("keyframes")]
    video_stats = {}
    for video_id, values in sorted(per_video_segments.items()):
        total = len(values)
        covered = sum(bool(item.get("keyframes")) for item in values)
        unique_count = len(per_video_frames.get(video_id, []))
        video_stats[video_id] = {
            "segments": total,
            "covered_segments": covered,
            "segment_coverage": covered / total,
            "unique_frames": unique_count,
            "zero_frame_segment_ids": [item["segment_id"] for item in values if not item.get("keyframes")],
        }

    report = {
        "definition": {
            "segment_coverage": "fraction of 20-second overlapping segments containing >=1 keyframe",
            "near_duplicate": "adjacent frames from one video with 64-bit DCT pHash Hamming distance <= 6",
        },
        "segments": len(segments),
        "videos": len(per_video_segments),
        "unique_frames": len(unique_frames),
        "frame_memberships": sum(memberships.values()),
        "reused_by_overlap": sum(count - 1 for count in memberships.values() if count > 1),
        "zero_frame_segments": len(zero_segments),
        "zero_frame_ratio": len(zero_segments) / len(segments),
        "zero_frame_segment_ids": zero_segments,
        "frame_count_histogram": dict(sorted(Counter(frame_counts).items())),
        "frames_per_segment_mean": mean(frame_counts),
        "frames_per_segment_median": median(frame_counts),
        "empty_searchable_segments": empty_searchable,
        "exact_duplicate_groups": sum(len(values) > 1 for values in exact_hashes.values()),
        "near_duplicate_adjacent_pairs": near_duplicate_pairs,
        "near_duplicate_pair_count": len(near_duplicate_pairs),
        "adjacent_phash_hamming_mean": mean(adjacent_distances) if adjacent_distances else None,
        "unreadable_frames": unreadable,
        "per_video": video_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"per_video", "zero_frame_segment_ids", "near_duplicate_adjacent_pairs"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
