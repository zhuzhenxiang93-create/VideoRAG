from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_VISUAL_PATTERNS = (
    r"画面(?:中|里|上)?",
    r"图(?:片|像)?(?:中|里|上)",
    r"什么颜色|哪种颜色|何种颜色",
    r"外观|标志|手势|动作|穿着|衣服",
    r"(?:看到|看见|显示|出现|举着|拿着|冲击)(?:了|的|什么|哪)",
)


@dataclass(frozen=True, slots=True)
class AdaptiveFusionPolicy:
    """Route-aware RRF weights selected from deterministic query intent cues."""

    sparse_source: str
    text_source: str
    vision_source: str
    sparse_weight: float = 1.00
    text_weight: float = 0.00
    vision_weight: float = 0.00
    visual_sparse_weight: float = 1.00
    visual_text_weight: float = 0.00
    visual_vision_weight: float = 2.00
    agreement_bonus: float = 0.05
    visual_patterns: tuple[str, ...] = field(default=DEFAULT_VISUAL_PATTERNS)

    def __post_init__(self) -> None:
        weights = (
            self.sparse_weight,
            self.text_weight,
            self.vision_weight,
            self.visual_sparse_weight,
            self.visual_text_weight,
            self.visual_vision_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Adaptive fusion weights must be non-negative")
        if sum(weights[:3]) <= 0 or sum(weights[3:]) <= 0:
            raise ValueError("Each adaptive fusion route must enable at least one source")
        if self.agreement_bonus < 0:
            raise ValueError("agreement_bonus must be non-negative")

    def intent(self, query: str) -> str:
        return (
            "visual"
            if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in self.visual_patterns)
            else "text"
        )

    def source_weights(self, query: str) -> dict[str, float]:
        if self.intent(query) == "visual":
            return {
                self.sparse_source: self.visual_sparse_weight,
                self.text_source: self.visual_text_weight,
                self.vision_source: self.visual_vision_weight,
            }
        return {
            self.sparse_source: self.sparse_weight,
            self.text_source: self.text_weight,
            self.vision_source: self.vision_weight,
        }
