from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TimedText:
    start_time: float
    end_time: float
    text: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start_time < 0 or self.end_time <= self.start_time:
            raise ValueError("TimedText requires 0 <= start_time < end_time")


@dataclass(frozen=True, slots=True)
class Keyframe:
    timestamp: float
    path: str
    caption: str = ""
    selection_source: str = "scene_change_v1"
    described_by_vlm: bool = True


@dataclass(frozen=True, slots=True)
class OCRText:
    timestamp: float
    text: str
    confidence: float
    bbox: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("OCRText timestamp must be non-negative")
        if not self.text.strip():
            raise ValueError("OCRText text must not be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("OCRText confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class VideoSegment:
    segment_id: str
    video_id: str
    source_path: str
    start_time: float
    end_time: float
    transcript: str = ""
    visual_caption: str = ""
    ocr_text: str = ""
    keyframes: tuple[Keyframe, ...] = field(default_factory=tuple)
    ocr_items: tuple[OCRText, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.segment_id or not self.video_id:
            raise ValueError("segment_id and video_id are required")
        if self.start_time < 0 or self.end_time <= self.start_time:
            raise ValueError("VideoSegment requires 0 <= start_time < end_time")

    @property
    def searchable_text(self) -> str:
        return "\n".join(
            value
            for value in (
                f"[ASR] {self.transcript.strip()}" if self.transcript.strip() else "",
                f"[视觉] {self.visual_caption.strip()}" if self.visual_caption.strip() else "",
            )
            if value
        )

    @property
    def evidence_text(self) -> str:
        return "\n".join(
            value
            for value in (
                self.searchable_text,
                f"[OCR] {self.ocr_text.strip()}" if self.ocr_text.strip() else "",
            )
            if value
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchHit:
    segment_id: str
    score: float
    source: str
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Evidence:
    segment: VideoSegment
    fused_score: float
    rerank_score: float

    def to_dict(self) -> dict[str, Any]:
        data = self.segment.to_dict()
        data.update(
            {
                "fused_score": self.fused_score,
                "rerank_score": self.rerank_score,
            }
        )
        return data


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    answerable: bool
    citations: tuple[str, ...] = field(default_factory=tuple)
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("GeneratedAnswer confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Answer:
    answer: str
    evidence: tuple[Evidence, ...]
    abstained: bool = False
    citations: tuple[str, ...] = field(default_factory=tuple)
    confidence: float | None = None
    route_labels: tuple[str, ...] = field(default_factory=tuple)
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "abstained": self.abstained,
            "citations": list(self.citations),
            "confidence": self.confidence,
            "route_labels": list(self.route_labels),
            "evidence": [item.to_dict() for item in self.evidence],
            "latency_ms": self.latency_ms,
        }
