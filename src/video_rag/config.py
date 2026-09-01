from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    strategy: str = "semantic"
    duration_seconds: float = 20.0
    overlap_seconds: float = 5.0
    minimum_seconds: float = 8.0
    maximum_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class OCRConfig:
    enabled: bool = True
    backend: str = "paddleocr"
    language: str = "ch"
    minimum_confidence: float = 0.55


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    strategy: str = "cascade"
    sparse_backend: str = "bm25"
    vision_backend: str = "chinese_clip"
    reranker_backend: str = "fusion_only"
    bm25_k1: float = 1.2
    bm25_b: float = 0.0
    adaptive_fusion: bool = True
    sparse_weight: float = 1.00
    text_weight: float = 0.00
    vision_weight: float = 0.00
    ocr_weight: float = 0.00
    visual_sparse_weight: float = 1.00
    visual_text_weight: float = 0.00
    visual_vision_weight: float = 2.00
    visual_ocr_weight: float = 0.0
    ocr_query_sparse_weight: float = 0.50
    ocr_query_text_weight: float = 0.00
    ocr_query_vision_weight: float = 0.50
    ocr_query_ocr_weight: float = 2.00
    agreement_bonus: float = 0.05
    reranker_weight: float = 0.00
    sparse_top_k: int = 20
    text_top_k: int = 20
    vision_top_k: int = 20
    ocr_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 3
    rrf_k: int = 60
    minimum_route_confidence: float = 0.70
    minimum_primary_score_margin: float = 0.0
    sparse_primary_min_score: float = 0.0
    text_primary_min_score: float = 0.20
    vision_primary_min_score: float = 0.20
    ocr_primary_min_score: float = 0.0
    neighbor_hops: int = 1
    temporal_neighbor_hops: int = 2
    dedupe_overlap_ratio: float = 0.20
    max_generation_segments: int = 8


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    backend: str = "qwen2_5_vl"
    evidence_mode: str = "images"
    max_images: int = 6
    max_frames: int = 16
    video_fps: float = 1.0
    minimum_rerank_score: float = 0.20
    minimum_generator_confidence: float = 0.20
    allow_abstention: bool = True
    require_citations: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    text_embedding: str
    reranker: str
    vision_language: str
    clip: str
    whisper: str
    qwen3_vl_embedding: str = "Qwen/Qwen3-VL-Embedding-2B"
    qwen3_vl_reranker: str = "Qwen/Qwen3-VL-Reranker-2B"
    qwen3_vl_generation: str = "Qwen/Qwen3-VL-4B-Instruct"
    qwen3_vl_repository: str = ""


@dataclass(frozen=True, slots=True)
class AppConfig:
    segmentation: SegmentationConfig
    ocr: OCRConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    models: ModelConfig


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    config = AppConfig(
        segmentation=SegmentationConfig(**raw.get("segmentation", {})),
        ocr=OCRConfig(**raw.get("ocr", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        models=ModelConfig(**raw["models"]),
    )
    allowed = {
        "segmentation.strategy": (
            config.segmentation.strategy,
            {"fixed", "semantic"},
        ),
        "ocr.backend": (config.ocr.backend, {"paddleocr"}),
        "retrieval.sparse_backend": (
            config.retrieval.sparse_backend,
            {"bm25", "bm25_like"},
        ),
        "retrieval.strategy": (
            config.retrieval.strategy,
            {"cascade", "rrf"},
        ),
        "retrieval.vision_backend": (
            config.retrieval.vision_backend,
            {"chinese_clip", "qwen3_vl"},
        ),
        "retrieval.reranker_backend": (
            config.retrieval.reranker_backend,
            {"fusion_only", "qwen3_text", "qwen3_vl"},
        ),
        "generation.backend": (
            config.generation.backend,
            {"qwen2_5_vl", "qwen3_vl"},
        ),
        "generation.evidence_mode": (
            config.generation.evidence_mode,
            {"images", "frame_sequence"},
        ),
    }
    for field, (value, choices) in allowed.items():
        if value not in choices:
            raise ValueError(f"{field} must be one of {sorted(choices)}, got {value!r}")
    if not 0 <= config.ocr.minimum_confidence <= 1:
        raise ValueError("ocr.minimum_confidence must be between 0 and 1")
    if not (
        0
        < config.segmentation.minimum_seconds
        <= config.segmentation.duration_seconds
        <= config.segmentation.maximum_seconds
    ):
        raise ValueError("segmentation requires 0 < minimum <= duration <= maximum")
    if not 0 <= config.segmentation.overlap_seconds < config.segmentation.minimum_seconds:
        raise ValueError("segmentation overlap must satisfy 0 <= overlap < minimum")
    if not 0 <= config.retrieval.dedupe_overlap_ratio <= 1:
        raise ValueError("retrieval.dedupe_overlap_ratio must be between 0 and 1")
    if not 0 <= config.retrieval.minimum_route_confidence <= 1:
        raise ValueError("retrieval.minimum_route_confidence must be between 0 and 1")
    if not 0 <= config.retrieval.minimum_primary_score_margin <= 1:
        raise ValueError("retrieval.minimum_primary_score_margin must be between 0 and 1")
    if min(
        config.retrieval.neighbor_hops,
        config.retrieval.temporal_neighbor_hops,
    ) < 0:
        raise ValueError("retrieval neighbor hops must be non-negative")
    return config
