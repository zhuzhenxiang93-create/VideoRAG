from __future__ import annotations

from collections.abc import Iterable

from video_rag.schemas import Keyframe, TimedText, VideoSegment


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


def materialize_segments(
    *,
    video_id: str,
    source_path: str,
    duration: float,
    transcript: Iterable[TimedText],
    keyframes: Iterable[Keyframe] = (),
    window_seconds: float = 20.0,
    overlap_seconds: float = 5.0,
) -> list[VideoSegment]:
    transcript_items = list(transcript)
    keyframe_items = list(keyframes)
    segments: list[VideoSegment] = []

    for index, (start, end) in enumerate(
        build_windows(duration, window_seconds, overlap_seconds)
    ):
        text = " ".join(
            item.text.strip()
            for item in transcript_items
            if item.text.strip() and _overlaps(start, end, item.start_time, item.end_time)
        )
        frames = tuple(frame for frame in keyframe_items if start <= frame.timestamp < end)
        caption = " ".join(frame.caption.strip() for frame in frames if frame.caption.strip())
        segments.append(
            VideoSegment(
                segment_id=f"{video_id}_{index:04d}",
                video_id=video_id,
                source_path=source_path,
                start_time=start,
                end_time=end,
                transcript=text,
                visual_caption=caption,
                keyframes=frames,
            )
        )
    return segments

