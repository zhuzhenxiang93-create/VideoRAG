from __future__ import annotations

import json
from pathlib import Path

from video_rag.schemas import Keyframe, OCRText, VideoSegment


def save_segments(path: str | Path, segments: list[VideoSegment]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for segment in segments:
            stream.write(json.dumps(segment.to_dict(), ensure_ascii=False) + "\n")


def load_segments(path: str | Path) -> list[VideoSegment]:
    segments: list[VideoSegment] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                data["keyframes"] = tuple(Keyframe(**item) for item in data.get("keyframes", ()))
                data["ocr_items"] = tuple(
                    OCRText(
                        **{
                            **item,
                            "bbox": tuple(tuple(point) for point in item.get("bbox", ())),
                        }
                    )
                    for item in data.get("ocr_items", ())
                )
                segments.append(VideoSegment(**data))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid segment on line {line_number}: {exc}") from exc
    return segments
