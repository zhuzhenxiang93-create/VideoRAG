from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
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


def probe_video_duration(path: Path) -> float | None:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps > 0 and frames > 0:
            return frames / fps
        return None
    finally:
        capture.release()


def percentile(values: list[float], q: float) -> float | None:
    """Linear percentile without requiring numpy in the reporting path."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def longest_zero_run(segments: list[dict]) -> dict:
    """Return the longest run of consecutive segments without a keyframe."""
    best: list[dict] = []
    current: list[dict] = []
    for segment in sorted(segments, key=lambda item: (item["start_time"], item["end_time"])):
        if segment.get("keyframes"):
            current = []
        else:
            current.append(segment)
            if len(current) > len(best):
                best = list(current)
    if not best:
        return {"count": 0, "start": None, "end": None, "span_seconds": 0.0, "segment_ids": []}
    return {
        "count": len(best),
        "start": float(best[0]["start_time"]),
        "end": float(best[-1]["end_time"]),
        "span_seconds": float(best[-1]["end_time"] - best[0]["start_time"]),
        "segment_ids": [item["segment_id"] for item in best],
    }


def plan_supplement(segments: list[dict], target_frames: int) -> dict:
    """Dry-run a deterministic, overlap-aware minimum-coverage frame plan.

    Candidate timestamps are chosen from the largest gap inside an under-covered
    segment.  One physical timestamp is then assigned to every overlapping
    segment that contains it, so shared windows do not cause duplicate encoding.
    This function plans timestamps only: it does not decode or write images.
    """
    if target_frames < 1:
        raise ValueError("target_frames must be >= 1")
    ordered = sorted(segments, key=lambda item: (item["start_time"], item["end_time"], item["segment_id"]))
    counts = {item["segment_id"]: len(item.get("keyframes", [])) for item in ordered}
    candidates: list[dict] = []
    while True:
        lacking = next((item for item in ordered if counts[item["segment_id"]] < target_frames), None)
        if lacking is None:
            break
        # Prefer the common intersection of under-covered overlapping windows,
        # because one decoded frame can then serve several segment memberships.
        group = [lacking]
        shared_start = float(lacking["start_time"])
        shared_end = float(lacking["end_time"])
        for item in ordered:
            if item is lacking or counts[item["segment_id"]] >= target_frames:
                continue
            next_start = max(shared_start, float(item["start_time"]))
            next_end = min(shared_end, float(item["end_time"]))
            if next_start < next_end:
                group.append(item)
                shared_start, shared_end = next_start, next_end
        timestamp = round((shared_start + shared_end) / 2.0, 3)
        memberships = [
            item["segment_id"]
            for item in ordered
            if float(item["start_time"]) <= timestamp < float(item["end_time"])
            and counts[item["segment_id"]] < target_frames
        ]
        if not memberships:
            memberships = [lacking["segment_id"]]
        candidates.append({"timestamp": timestamp, "segment_ids": memberships})
        for segment_id in memberships:
            counts[segment_id] += 1
    initial_memberships = sum(len(item.get("keyframes", [])) for item in ordered)
    new_memberships = sum(len(item["segment_ids"]) for item in candidates)
    return {
        "target_frames_per_segment": target_frames,
        "new_physical_frames": len(candidates),
        "new_memberships": new_memberships,
        "overlap_reused_memberships": new_memberships - len(candidates),
        "clip_encodes_required": len(candidates),
        "qwen_vl_calls_required": 0,
        "projected_physical_index_entries": len(candidates),
        "initial_memberships": initial_memberships,
        "candidate_timestamps": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit keyframe coverage and duplication.")
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/evaluation/keyframe_audit.json"), type=Path)
    parser.add_argument("--csv-output", default=None, type=Path)
    parser.add_argument("--boundary-tolerance", default=2.0, type=float)
    parser.add_argument("--estimated-jpeg-bytes", default=120_000, type=int)
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
    all_gap_values_by_video: dict[str, list[float]] = {}
    for video_id, values in sorted(per_video_segments.items()):
        values.sort(key=lambda item: (item["start_time"], item["end_time"]))
        total = len(values)
        covered = sum(bool(item.get("keyframes")) for item in values)
        frame_times = sorted(float(item[0]) for item in per_video_frames.get(video_id, []))
        unique_count = len(frame_times)
        video_start = min(float(item["start_time"]) for item in values)
        segment_end = max(float(item["end_time"]) for item in values)
        source_path = Path(values[0]["source_path"])
        measured_duration = probe_video_duration(source_path) if source_path.is_file() else None
        video_end = video_start + measured_duration if measured_duration is not None else segment_end
        internal_gaps = [right - left for left, right in zip(frame_times, frame_times[1:])]
        all_gaps = []
        if frame_times:
            all_gaps = [frame_times[0] - video_start, *internal_gaps, video_end - frame_times[-1]]
        else:
            all_gaps = [video_end - video_start]
        all_gap_values_by_video[video_id] = all_gaps
        zero_values = [item for item in values if not item.get("keyframes")]
        boundary_reuse = []
        for segment in zero_values:
            nearest = min(
                (min(abs(timestamp - float(segment["start_time"])), abs(timestamp - float(segment["end_time"]))) for timestamp in frame_times),
                default=None,
            )
            if nearest is not None and nearest <= args.boundary_tolerance:
                boundary_reuse.append({"segment_id": segment["segment_id"], "nearest_boundary_seconds": nearest})
        strategy_a = plan_supplement(values, 1)
        strategy_b = plan_supplement(values, 2)
        video_stats[video_id] = {
            "duration_seconds": video_end - video_start,
            "duration_source": "decoded frame_count/fps" if measured_duration is not None else "segment boundary fallback",
            "last_segment_end_seconds": segment_end,
            "segments": total,
            "covered_segments": covered,
            "segment_coverage": covered / total,
            "unique_frames": unique_count,
            "frames_per_minute": unique_count / ((video_end - video_start) / 60.0) if video_end > video_start else None,
            "zero_frame_segment_ids": [item["segment_id"] for item in zero_values],
            "zero_frame_ratio": len(zero_values) / total,
            "frame_gap_seconds": {
                "leading": frame_times[0] - video_start if frame_times else video_end - video_start,
                "trailing": video_end - frame_times[-1] if frame_times else video_end - video_start,
                "internal_max": max(internal_gaps, default=None),
                "max_including_boundaries": max(all_gaps),
                "p50_including_boundaries": percentile(all_gaps, 0.50),
                "p90_including_boundaries": percentile(all_gaps, 0.90),
                "p95_including_boundaries": percentile(all_gaps, 0.95),
            },
            "longest_consecutive_zero_segments": longest_zero_run(values),
            "boundary_nearby_but_outside_membership": boundary_reuse,
            "supplement_strategy_a": {key: value for key, value in strategy_a.items() if key != "candidate_timestamps"},
            "supplement_strategy_b": {key: value for key, value in strategy_b.items() if key != "candidate_timestamps"},
        }

    global_gaps = [gap for values in all_gap_values_by_video.values() for gap in values]

    strategy_a_videos = {video_id: plan_supplement(values, 1) for video_id, values in per_video_segments.items()}
    strategy_b_videos = {video_id: plan_supplement(values, 2) for video_id, values in per_video_segments.items()}

    def aggregate_plan(plans: dict[str, dict], target: int) -> dict:
        physical = sum(item["new_physical_frames"] for item in plans.values())
        memberships_count = sum(item["new_memberships"] for item in plans.values())
        return {
            "target_frames_per_segment": target,
            "new_physical_frames": physical,
            "new_memberships": memberships_count,
            "overlap_reused_memberships": memberships_count - physical,
            "clip_encodes_required": physical,
            "qwen_vl_calls_required": 0,
            "estimated_jpeg_bytes": physical * args.estimated_jpeg_bytes,
            "estimated_float32_vector_bytes": physical * 512 * 4,
            "projected_total_physical_frames": len(unique_frames) + physical,
            "method": "deterministic largest-gap timestamps; actual frames require decode, cheap dedupe and validation",
        }

    report = {
        "definition": {
            "segment_coverage": "fraction of 20-second overlapping segments containing >=1 keyframe",
            "near_duplicate": "adjacent frames from one video with 64-bit DCT pHash Hamming distance <= 6",
            "video_duration": "decoded frame_count/fps when readable; final segment boundary is an explicit fallback",
            "frame_gap": "seconds between adjacent frames, including leading/trailing segment-derived boundaries",
            "boundary_nearby": f"existing frame outside a zero-frame segment but <= {args.boundary_tolerance}s from a boundary; diagnostic only, not assigned",
            "supplement_dry_run": "overlap-aware timestamps only; no frames decoded and no index modified",
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
        "empty_searchable_segment_details": [
            {
                "segment_id": item["segment_id"],
                "keyframe_count": len(item.get("keyframes", [])),
                "vision_indexable": bool(item.get("keyframes")),
            }
            for item in segments
            if item["segment_id"] in empty_searchable
        ],
        "exact_duplicate_groups": sum(len(values) > 1 for values in exact_hashes.values()),
        "near_duplicate_adjacent_pairs": near_duplicate_pairs,
        "near_duplicate_pair_count": len(near_duplicate_pairs),
        "adjacent_phash_hamming_mean": mean(adjacent_distances) if adjacent_distances else None,
        "adjacent_phash_hamming_distribution": dict(sorted(Counter(adjacent_distances).items())),
        "frame_gap_seconds_all_videos": {
            "max": max(global_gaps),
            "p50": percentile(global_gaps, 0.50),
            "p90": percentile(global_gaps, 0.90),
            "p95": percentile(global_gaps, 0.95),
        },
        "max_consecutive_zero_segments": max(
            (item["longest_consecutive_zero_segments"]["count"] for item in video_stats.values()), default=0
        ),
        "boundary_nearby_zero_segments": sum(
            len(item["boundary_nearby_but_outside_membership"]) for item in video_stats.values()
        ),
        "supplement_dry_run": {
            "strategy_a_zero_to_one": aggregate_plan(strategy_a_videos, 1),
            "strategy_b_all_to_two": aggregate_plan(strategy_b_videos, 2),
        },
        "unreadable_frames": unreadable,
        "per_video": video_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_output = args.csv_output or args.output.with_suffix(".per_video.csv")
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "video_id", "duration_seconds", "segments", "unique_frames", "frames_per_minute",
            "zero_segments", "zero_ratio", "max_gap_seconds", "max_zero_run",
            "strategy_a_new_frames", "strategy_b_new_frames",
        ])
        writer.writeheader()
        for video_id, item in video_stats.items():
            writer.writerow({
                "video_id": video_id,
                "duration_seconds": item["duration_seconds"],
                "segments": item["segments"],
                "unique_frames": item["unique_frames"],
                "frames_per_minute": item["frames_per_minute"],
                "zero_segments": len(item["zero_frame_segment_ids"]),
                "zero_ratio": item["zero_frame_ratio"],
                "max_gap_seconds": item["frame_gap_seconds"]["max_including_boundaries"],
                "max_zero_run": item["longest_consecutive_zero_segments"]["count"],
                "strategy_a_new_frames": item["supplement_strategy_a"]["new_physical_frames"],
                "strategy_b_new_frames": item["supplement_strategy_b"]["new_physical_frames"],
            })
    print(json.dumps({key: value for key, value in report.items() if key not in {"per_video", "zero_frame_segment_ids", "near_duplicate_adjacent_pairs"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
