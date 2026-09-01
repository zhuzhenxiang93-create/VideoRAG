from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video_rag.adapters.qwen3_vl import _existing_images, _load_official_class
from video_rag.retrieval.faiss_dense import FaissDenseRetriever
from video_rag.schemas import VideoSegment


class Qwen3VLEmbeddingRetriever(FaissDenseRetriever):
    """Joint text-and-keyframe segment retrieval using Qwen3-VL-Embedding."""

    name = "vision_multimodal_qwen3_vl"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        *,
        implementation_repository: str | Path,
        index_dir: str | Path | None = None,
        force_rebuild: bool = False,
        instruction: str = "Retrieve video segments that answer the user's question.",
        max_frames: int = 16,
        fps: float = 1.0,
        batch_size: int = 2,
        model_factory: Any = None,
    ) -> None:
        super().__init__(index_dir, force_rebuild=force_rebuild)
        self.model_name = model_name
        self.implementation_repository = Path(implementation_repository)
        self.instruction = instruction
        self.max_frames = max_frames
        self.fps = fps
        self.batch_size = batch_size
        self._model_factory = model_factory
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            model_class = self._model_factory or _load_official_class(
                self.implementation_repository,
                "qwen3_vl_embedding",
                "Qwen3VLEmbedder",
            )
            self._model = model_class(model_name_or_path=self.model_name)
        return self._model

    @staticmethod
    def _numpy(vectors: Any) -> np.ndarray:
        if hasattr(vectors, "detach"):
            vectors = vectors.detach().float().cpu().numpy()
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, segments: list[VideoSegment]) -> tuple[np.ndarray, list[str]]:
        inputs: list[dict[str, Any]] = []
        identifiers: list[str] = []
        for segment in segments:
            item: dict[str, Any] = {
                "text": (
                    f"Time {segment.start_time:.3f}-{segment.end_time:.3f} seconds\n"
                    f"{segment.searchable_text}"
                )
            }
            images = _existing_images(segment, limit=self.max_frames)
            if images:
                item["video"] = images
            if not item["text"].strip() and not images:
                continue
            inputs.append(item)
            identifiers.append(segment.segment_id)

        batches: list[np.ndarray] = []
        model = self._load()
        for start in range(0, len(inputs), self.batch_size):
            batches.append(self._numpy(model.process(inputs[start : start + self.batch_size])))
        if not batches:
            return np.empty((0, 0), dtype=np.float32), []
        return np.concatenate(batches, axis=0), identifiers

    def encode_query(self, query: str) -> np.ndarray:
        return self._numpy(
            self._load().process(
                [{"text": query, "instruction": self.instruction}]
            )
        )
