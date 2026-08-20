from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path

from video_rag.adapters import QwenVLCaptioner, QwenVLService
from video_rag.config import load_config
from video_rag.ingestion import (
    SceneKeyframeExtractor,
    WhisperTranscriber,
    materialize_segments,
    probe_video,
)
from video_rag.schemas import Keyframe
from video_rag.storage import save_segments

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".ogv", ".avi", ".m4v"}


def video_identifier(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    safe_stem = "".join(char if char.isalnum() else "_" for char in path.stem)
    return f"{safe_stem}_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create timestamped multimodal video segments.")
    parser.add_argument("--input", required=True, type=Path, help="Video file or directory")
    parser.add_argument("--output", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--frames-dir", default=Path("artifacts/frames"), type=Path)
    parser.add_argument("--config", default=Path("config.toml"), type=Path)
    parser.add_argument("--language", default=None)
    parser.add_argument("--skip-captions", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    videos = (
        [args.input]
        if args.input.is_file()
        else sorted(path for path in args.input.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    )
    if not videos:
        raise SystemExit("No supported videos found")

    whisper_name = config.models.whisper
    if "/" not in whisper_name:
        whisper_name = f"openai/whisper-{whisper_name}"
    transcriber = WhisperTranscriber(whisper_name)
    extractor = SceneKeyframeExtractor()
    prepared: list[tuple[Path, str, float, list, list[Keyframe]]] = []

    for video in videos:
        identifier = video_identifier(video)
        info = probe_video(video)
        print(f"[ASR] {video.name}")
        transcript = transcriber.transcribe(video, language=args.language)
        print(f"[Frames] {video.name}")
        frames = extractor.extract(video, args.frames_dir / identifier)
        prepared.append((video.resolve(), identifier, info.duration, transcript, frames))
    transcriber.unload()

    if not args.skip_captions:
        service = QwenVLService(config.models.vision_language)
        captioner = QwenVLCaptioner(service)
        captioned: list[tuple[Path, str, float, list, list[Keyframe]]] = []
        for video, identifier, duration, transcript, frames in prepared:
            enriched: list[Keyframe] = []
            for frame in frames:
                print(f"[Caption] {Path(frame.path).name}")
                enriched.append(replace(frame, caption=captioner.caption(frame)))
            captioned.append((video, identifier, duration, transcript, enriched))
        service.unload()
        prepared = captioned

    segments = []
    for video, identifier, duration, transcript, frames in prepared:
        segments.extend(
            materialize_segments(
                video_id=identifier,
                source_path=str(video),
                duration=duration,
                transcript=transcript,
                keyframes=frames,
                window_seconds=config.segmentation.duration_seconds,
                overlap_seconds=config.segmentation.overlap_seconds,
            )
        )
    save_segments(args.output, segments)
    print(f"Saved {len(segments)} segments to {args.output}")


if __name__ == "__main__":
    main()
