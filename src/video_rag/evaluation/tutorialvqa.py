from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import mean, median

from video_rag.schemas import VideoSegment


def overlaps_reference(segment: VideoSegment, intervals: Sequence[dict]) -> bool:
    return any(
        max(segment.start_time, float(interval["start_time"]))
        < min(segment.end_time, float(interval["end_time"]))
        for interval in intervals
    )


def relevant_segment_ids(
    segments: Iterable[VideoSegment], video_id: str, intervals: Sequence[dict]
) -> list[str]:
    return [
        segment.segment_id
        for segment in segments
        if segment.video_id == video_id and overlaps_reference(segment, intervals)
    ]


def interval_iou(start: float, end: float, reference_start: float, reference_end: float) -> float:
    intersection = max(0.0, min(end, reference_end) - max(start, reference_start))
    union = max(end, reference_end) - min(start, reference_start)
    return intersection / union if union > 0 else 0.0


def best_temporal_iou(segments: Sequence[VideoSegment], intervals: Sequence[dict]) -> float:
    return max(
        (
            interval_iou(
                segment.start_time,
                segment.end_time,
                float(interval["start_time"]),
                float(interval["end_time"]),
            )
            for segment in segments
            for interval in intervals
        ),
        default=0.0,
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
    }
