from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    duration_seconds: float = 20.0
    overlap_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    sparse_backend: str = "bm25"
    vision_backend: str = "chinese_clip"
    reranker_backend: str = "qwen3_text"
    sparse_top_k: int = 20
    text_top_k: int = 20
    vision_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 3
    rrf_k: int = 60


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    backend: str = "qwen2_5_vl"
    evidence_mode: str = "images"
    max_images: int = 6
    max_frames: int = 16
    video_fps: float = 1.0
    minimum_rerank_score: float = 0.20
    allow_abstention: bool = True


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
    retrieval: RetrievalConfig
    generation: GenerationConfig
    models: ModelConfig


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    config = AppConfig(
        segmentation=SegmentationConfig(**raw.get("segmentation", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        models=ModelConfig(**raw["models"]),
    )
    allowed = {
        "retrieval.sparse_backend": (
            config.retrieval.sparse_backend,
            {"bm25", "bm25_like"},
        ),
        "retrieval.vision_backend": (
            config.retrieval.vision_backend,
            {"chinese_clip", "qwen3_vl"},
        ),
        "retrieval.reranker_backend": (
            config.retrieval.reranker_backend,
            {"qwen3_text", "qwen3_vl"},
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
    return config
