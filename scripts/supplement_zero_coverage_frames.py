from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from scripts.audit_keyframes import hamming, perceptual_hash, plan_supplement


def image_phash(image) -> int:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    values = cv2.dct(np.float32(resized))[:8, :8]
    threshold = float(np.median(values[1:, :]))
    result = 0
    for bit in (values > threshold).flatten():
        result = (result << 1) | int(bit)
    return result


def decode_at(path: Path, timestamp: float):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise ValueError(f"invalid FPS for {path}")
        frame_index = max(0, round(timestamp * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        if not ok or image is None:
            raise ValueError(f"cannot decode {path} at {timestamp:.3f}s")
        return image, frame_index / fps
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplement only zero-coverage segments without VLM captioning.")
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.jsonl"))
    parser.add_argument("--output-segments", type=Path, default=Path("artifacts/segments.supplement_a.jsonl"))
    parser.add_argument("--frame-dir", type=Path, default=Path("artifacts/frames_supplement_a"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation/supplement_a_build.json"))
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_segments.exists() and not args.force:
        raise ValueError(f"{args.output_segments} already exists; use --force only for an intentional rebuild")

    segments = [json.loads(line) for line in args.segments.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_video: defaultdict[str, list[dict]] = defaultdict(list)
    for segment in segments:
        by_video[segment["video_id"]].append(segment)
    before_zero = sum(not item.get("keyframes") for item in segments)
    existing_hashes: defaultdict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for segment in segments:
        for frame in segment.get("keyframes", []):
            path = Path(frame["path"])
            key = (segment["video_id"], path.as_posix())
            if not any(entry[2] == key[1] for entry in existing_hashes[key[0]]) and path.is_file():
                existing_hashes[key[0]].append((perceptual_hash(path), float(frame["timestamp"]), path.as_posix()))

    generated = []
    for video_id, video_segments in sorted(by_video.items()):
        plan = plan_supplement(video_segments, 1)
        segment_by_id = {item["segment_id"]: item for item in video_segments}
        for ordinal, candidate in enumerate(plan["candidate_timestamps"]):
            target_segments = [segment_by_id[item] for item in candidate["targeted_segment_ids"]]
            allowed_start = max(float(item["start_time"]) for item in target_segments)
            allowed_end = min(float(item["end_time"]) for item in target_segments)
            fractions = (0.5, 0.35, 0.65, 0.2, 0.8)
            trials = []
            source = Path(video_segments[0]["source_path"])
            for fraction in fractions:
                requested = allowed_start + (allowed_end - allowed_start) * fraction
                image, actual = decode_at(source, requested)
                value = image_phash(image)
                distances = [hamming(value, known[0]) for known in existing_hashes[video_id]]
                nearest = min(distances, default=64)
                trials.append((nearest, actual, requested, value, image))
                if nearest > args.phash_threshold:
                    break
            nearest, actual, requested, value, image = max(trials, key=lambda item: item[0])
            memberships = [
                item for item in video_segments
                if float(item["start_time"]) <= actual < float(item["end_time"])
            ]
            if not all(item in memberships for item in target_segments):
                raise ValueError(f"decoded timestamp {actual} left target interval for {video_id}")
            directory = args.frame_dir / video_id
            directory.mkdir(parents=True, exist_ok=True)
            output_path = directory / f"supplement_a_{ordinal:04d}_{actual:.3f}.jpg"
            import cv2
            if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise ValueError(f"failed to write {output_path}")
            frame_record = {
                "timestamp": actual,
                "path": output_path.as_posix(),
                "caption": "",
                "selection_source": "zero_coverage_supplement_v1",
                "described_by_vlm": False,
            }
            for segment in memberships:
                segment.setdefault("keyframes", []).append(dict(frame_record))
                segment["keyframes"].sort(key=lambda item: (item["timestamp"], item["path"]))
            near_duplicate = nearest <= args.phash_threshold
            existing_hashes[video_id].append((value, actual, output_path.as_posix()))
            generated.append({
                "video_id": video_id,
                "requested_timestamp": requested,
                "actual_timestamp": actual,
                "frame_path": output_path.as_posix(),
                "segment_ids": [item["segment_id"] for item in memberships],
                "targeted_segment_ids": candidate["targeted_segment_ids"],
                "nearest_phash_hamming": nearest,
                "near_duplicate_retained_for_temporal_coverage": near_duplicate,
                "selection_attempts": len(trials),
            })

    after_zero = sum(not item.get("keyframes") for item in segments)
    if after_zero != 0:
        raise ValueError(f"strategy A failed: {after_zero} segments still have no frame")
    args.output_segments.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in segments)
    args.output_segments.write_text(content, encoding="utf-8")
    report = {
        "schema_version": 1,
        "strategy": "A: only zero-frame segments are targeted to reach >=1 frame",
        "source_segments": str(args.segments),
        "source_segments_sha256": hashlib.sha256(args.segments.read_bytes()).hexdigest(),
        "output_segments": str(args.output_segments),
        "output_segments_sha256": hashlib.sha256(args.output_segments.read_bytes()).hexdigest(),
        "strict_membership_policy": "start <= actual_timestamp < end",
        "phash": {"algorithm": "64-bit DCT pHash", "threshold": args.phash_threshold},
        "zero_segments_before": before_zero,
        "zero_segments_after": after_zero,
        "new_physical_frames": len(generated),
        "new_memberships": sum(len(item["segment_ids"]) for item in generated),
        "target_coverage_gains": sum(len(item["targeted_segment_ids"]) for item in generated),
        "near_duplicates_retained_for_temporal_coverage": sum(item["near_duplicate_retained_for_temporal_coverage"] for item in generated),
        "qwen_vl_calls": 0,
        "generated_frames": generated,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "generated_frames"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
