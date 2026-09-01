from __future__ import annotations

from collections.abc import Iterable

from video_rag.schemas import Keyframe, OCRText, TimedText, VideoSegment


def build_windows(
    video_duration: float,
    window_seconds: float = 20.0,
    overlap_seconds: float = 5.0,
) -> list[tuple[float, float]]:
    if video_duration <= 0:
        raise ValueError("video_duration must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= window_seconds:
        raise ValueError("overlap_seconds must satisfy 0 <= overlap < window")

    windows: list[tuple[float, float]] = []
    step = window_seconds - overlap_seconds
    start = 0.0
    while start < video_duration:
        end = min(start + window_seconds, video_duration)
        windows.append((round(start, 3), round(end, 3)))
        if end >= video_duration:
            break
        start += step
    return windows


def _overlaps(start: float, end: float, item_start: float, item_end: float) -> bool:
    return item_start < end and item_end > start


def build_semantic_windows(
    video_duration: float,
    transcript: Iterable[TimedText],
    keyframes: Iterable[Keyframe] = (),
    *,
    target_seconds: float = 20.0,
    overlap_seconds: float = 5.0,
    minimum_seconds: float = 8.0,
    maximum_seconds: float = 30.0,
) -> list[tuple[float, float]]:
    """Choose window ends near ASR or scene boundaries instead of arbitrary timestamps."""
    if not 0 < minimum_seconds <= target_seconds <= maximum_seconds:
        raise ValueError("semantic windows require 0 < minimum <= target <= maximum")
    if overlap_seconds < 0 or overlap_seconds >= minimum_seconds:
        raise ValueError("semantic overlap must satisfy 0 <= overlap < minimum")
    if video_duration <= 0:
        raise ValueError("video_duration must be positive")

    transcript_boundaries = {
        round(item.end_time, 3)
        for item in transcript
        if 0 < item.end_time < video_duration
    }
    scene_boundaries = {
        round(frame.timestamp, 3)
        for frame in keyframes
        if 0 < frame.timestamp < video_duration
    }
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < video_duration:
        earliest = min(video_duration, start + minimum_seconds)
        latest = min(video_duration, start + maximum_seconds)
        target = min(video_duration, start + target_seconds)
        if latest >= video_duration:
            end = video_duration
        else:
            candidates = [
                (boundary, 0 if boundary in transcript_boundaries else 1)
                for boundary in transcript_boundaries | scene_boundaries
                if earliest <= boundary <= latest
            ]
            end = (
                min(candidates, key=lambda item: (abs(item[0] - target), item[1], item[0]))[0]
                if candidates
                else latest
            )
        end = round(end, 3)
        windows.append((round(start, 3), end))
        if end >= video_duration:
            break
        next_start = round(end - overlap_seconds, 3)
        if next_start <= start:
            next_start = round(start + minimum_seconds - overlap_seconds, 3)
        start = next_start
    return windows


def materialize_segments(
    *,
    video_id: str,
    source_path: str,
    duration: float,
    transcript: Iterable[TimedText],
    keyframes: Iterable[Keyframe] = (),
    ocr_items: Iterable[OCRText] = (),
    window_seconds: float = 20.0,
    overlap_seconds: float = 5.0,
    strategy: str = "fixed",
    minimum_seconds: float = 8.0,
    maximum_seconds: float = 30.0,
) -> list[VideoSegment]:
    transcript_items = list(transcript)
    keyframe_items = list(keyframes)
    ocr_values = list(ocr_items)
    segments: list[VideoSegment] = []

    if strategy == "semantic":
        windows = build_semantic_windows(
            duration,
            transcript_items,
            keyframe_items,
            target_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            minimum_seconds=minimum_seconds,
            maximum_seconds=maximum_seconds,
        )
    elif strategy == "fixed":
        windows = build_windows(duration, window_seconds, overlap_seconds)
    else:
        raise ValueError("strategy must be 'fixed' or 'semantic'")

    for index, (start, end) in enumerate(windows):
        text = " ".join(
            item.text.strip()
            for item in transcript_items
            if item.text.strip() and _overlaps(start, end, item.start_time, item.end_time)
        )
        frames = tuple(frame for frame in keyframe_items if start <= frame.timestamp < end)
        segment_ocr = tuple(item for item in ocr_values if start <= item.timestamp < end)
        caption = " ".join(frame.caption.strip() for frame in frames if frame.caption.strip())
        ocr_text = " ".join(dict.fromkeys(item.text.strip() for item in segment_ocr))
        segments.append(
            VideoSegment(
                segment_id=f"{video_id}_{index:04d}",
                video_id=video_id,
                source_path=source_path,
                start_time=start,
                end_time=end,
                transcript=text,
                visual_caption=caption,
                ocr_text=ocr_text,
                keyframes=frames,
                ocr_items=segment_ocr,
            )
        )
    return segments
