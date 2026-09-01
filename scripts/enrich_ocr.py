from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from video_rag.config import load_config
from video_rag.ingestion import PaddleOCRExtractor
from video_rag.schemas import Keyframe, OCRText, VideoSegment
from video_rag.storage import load_segments, save_segments


def enrich_segments(
    segments: list[VideoSegment], extractor: PaddleOCRExtractor
) -> list[VideoSegment]:
    frames_by_video: dict[str, dict[str, Keyframe]] = {}
    for segment in segments:
        video_frames = frames_by_video.setdefault(segment.video_id, {})
        for frame in segment.keyframes:
            video_frames.setdefault(frame.path, frame)

    ocr_by_video: dict[str, list[OCRText]] = {}
    for video_id, frame_map in frames_by_video.items():
        frames = sorted(frame_map.values(), key=lambda item: (item.timestamp, item.path))
        print(f"[OCR] {video_id}: {len(frames)} unique keyframes")
        ocr_by_video[video_id] = extractor.extract(frames)

    enriched: list[VideoSegment] = []
    for segment in segments:
        items = tuple(
            item
            for item in ocr_by_video.get(segment.video_id, ())
            if segment.start_time <= item.timestamp < segment.end_time
        )
        text = " ".join(dict.fromkeys(item.text.strip() for item in items))
        enriched.append(replace(segment, ocr_text=text, ocr_items=items))
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add timestamped OCR to an existing segment file without rerunning ASR."
    )
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/segments.ocr.jsonl")
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()
    if args.output.resolve() == args.segments.resolve():
        raise ValueError("refusing to overwrite the source segment file; use a new --output")

    config = load_config(args.config)
    extractor = PaddleOCRExtractor(
        language=config.ocr.language,
        minimum_confidence=config.ocr.minimum_confidence,
    )
    segments = load_segments(args.segments)
    enriched = enrich_segments(segments, extractor)
    save_segments(args.output, enriched)
    recognized_segments = sum(bool(segment.ocr_text) for segment in enriched)
    print(
        f"Saved {len(enriched)} segments to {args.output}; "
        f"OCR text present in {recognized_segments} segments"
    )


if __name__ == "__main__":
    main()
