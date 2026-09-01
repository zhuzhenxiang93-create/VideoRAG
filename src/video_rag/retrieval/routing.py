from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_VISUAL_PATTERNS = (
    r"画面(?:中|里|上)?",
    r"图(?:片|像)?(?:中|里|上)",
    r"什么颜色|哪种颜色|何种颜色",
    r"外观|标志|手势|动作|穿着|衣服",
    r"(?:看到|看见|出现|举着|拿着|冲击)(?:了|的|什么|哪)",
)
DEFAULT_OCR_PATTERNS = (
    r"OCR|字幕(?:上|中|里)?(?:写|显示|出现|是)",
    r"屏幕(?:上|中|里)?.*(?:文字|数字|写着|显示)",
    r"(?:标题|标牌|横幅|右下角|左下角|右上角|左上角).*(?:文字|字样|数字|名称|写)",
    r"写着什么|显示的(?:文字|数字|名称)|文字内容",
)
DEFAULT_MULTIMODAL_PATTERNS = (
    r"结合(?:画面|图表|字幕|解说|语音)",
    r"(?:画面|图表|字幕).*(?:解说|提到|说)",
    r"(?:解说|语音).*(?:画面|图表|字幕)",
)
DEFAULT_TEMPORAL_PATTERNS = (
    r"随后|之后|之前|接下来|最初|最后",
    r"先.*再|顺序|过程|前后|发生了什么变化",
)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    labels: tuple[str, ...]
    source_weights: dict[str, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class AdaptiveFusionPolicy:
    """Deterministic multi-label routing with separately calibrated modality weights."""

    sparse_source: str
    text_source: str
    vision_source: str
    ocr_source: str | None = None
    sparse_weight: float = 1.00
    text_weight: float = 0.00
    vision_weight: float = 0.00
    ocr_weight: float = 0.00
    visual_sparse_weight: float = 1.00
    visual_text_weight: float = 0.00
    visual_vision_weight: float = 2.00
    visual_ocr_weight: float = 0.25
    ocr_query_sparse_weight: float = 0.50
    ocr_query_text_weight: float = 0.00
    ocr_query_vision_weight: float = 0.50
    ocr_query_ocr_weight: float = 2.00
    agreement_bonus: float = 0.05
    visual_patterns: tuple[str, ...] = field(default=DEFAULT_VISUAL_PATTERNS)
    ocr_patterns: tuple[str, ...] = field(default=DEFAULT_OCR_PATTERNS)
    multimodal_patterns: tuple[str, ...] = field(default=DEFAULT_MULTIMODAL_PATTERNS)
    temporal_patterns: tuple[str, ...] = field(default=DEFAULT_TEMPORAL_PATTERNS)

    def __post_init__(self) -> None:
        weights = (
            self.sparse_weight,
            self.text_weight,
            self.vision_weight,
            self.ocr_weight,
            self.visual_sparse_weight,
            self.visual_text_weight,
            self.visual_vision_weight,
            self.visual_ocr_weight,
            self.ocr_query_sparse_weight,
            self.ocr_query_text_weight,
            self.ocr_query_vision_weight,
            self.ocr_query_ocr_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Adaptive fusion weights must be non-negative")
        if sum(weights[:4]) <= 0 or sum(weights[4:8]) <= 0 or sum(weights[8:]) <= 0:
            raise ValueError("Each adaptive fusion route must enable at least one source")
        if self.agreement_bonus < 0:
            raise ValueError("agreement_bonus must be non-negative")

    @staticmethod
    def _matches(query: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns)

    def intents(self, query: str) -> tuple[str, ...]:
        labels: list[str] = []
        if self._matches(query, self.ocr_patterns):
            labels.append("ocr")
        if self._matches(query, self.visual_patterns):
            labels.append("visual")
        if self._matches(query, self.multimodal_patterns):
            labels.append("multimodal")
        if self._matches(query, self.temporal_patterns):
            labels.append("temporal")
        if not labels:
            labels.append("text")
        return tuple(labels)

    def intent(self, query: str) -> str:
        labels = self.intents(query)
        for preferred in ("multimodal", "ocr", "visual", "temporal", "text"):
            if preferred in labels:
                return preferred
        return "text"

    def decision(self, query: str) -> RoutingDecision:
        labels = self.intents(query)
        text_route = (
            self.sparse_weight,
            self.text_weight,
            self.vision_weight,
            self.ocr_weight,
        )
        visual_route = (
            self.visual_sparse_weight,
            self.visual_text_weight,
            self.visual_vision_weight,
            self.visual_ocr_weight,
        )
        ocr_route = (
            self.ocr_query_sparse_weight,
            self.ocr_query_text_weight,
            self.ocr_query_vision_weight,
            self.ocr_query_ocr_weight,
        )
        if "multimodal" in labels:
            routes = [text_route, visual_route, ocr_route]
        else:
            routes = []
            if "visual" in labels:
                routes.append(visual_route)
            if "ocr" in labels:
                routes.append(ocr_route)
            if not routes:
                routes.append(text_route)
        combined = tuple(max(values) for values in zip(*routes, strict=True))
        sources = (self.sparse_source, self.text_source, self.vision_source, self.ocr_source)
        weights = {
            source: weight
            for source, weight in zip(sources, combined, strict=True)
            if source is not None
        }
        confidence = 0.95 if "multimodal" in labels else 0.90 if labels != ("text",) else 0.75
        return RoutingDecision(labels, weights, confidence)

    def source_weights(self, query: str) -> dict[str, float]:
        return self.decision(query).source_weights
