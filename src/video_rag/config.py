from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    duration_seconds: float = 20.0
    overlap_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    sparse_top_k: int = 20
    text_top_k: int = 20
    vision_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 3
    rrf_k: int = 60


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    minimum_rerank_score: float = 0.20
    allow_abstention: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    text_embedding: str
    reranker: str
    vision_language: str
    clip: str
    whisper: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    segmentation: SegmentationConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    models: ModelConfig


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return AppConfig(
        segmentation=SegmentationConfig(**raw.get("segmentation", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        models=ModelConfig(**raw["models"]),
    )

