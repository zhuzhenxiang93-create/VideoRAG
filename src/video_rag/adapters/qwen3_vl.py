from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from video_rag.schemas import VideoSegment


def _load_official_class(repository: str | Path, module: str, class_name: str) -> type[Any]:
    """Load a class from an explicitly cloned official Qwen3-VL-Embedding repository."""
    root = Path(repository).expanduser().resolve()
    expected = root / "src" / "models" / f"{module.rsplit('.', 1)[-1]}.py"
    if not expected.exists():
        raise RuntimeError(
            "Qwen3-VL-Embedding implementation repository is missing. "
            f"Expected {expected}. Clone https://github.com/QwenLM/Qwen3-VL-Embedding "
            "and set models.qwen3_vl_repository in the config."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        imported = importlib.import_module(f"src.models.{module}")
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import the official Qwen3-VL implementation. "
            "Install its dependencies with `pip install -e <repository>`."
        ) from exc
    return getattr(imported, class_name)


def _existing_images(segment: VideoSegment, *, limit: int | None = None) -> list[str]:
    paths = [frame.path for frame in segment.keyframes if Path(frame.path).exists()]
    return paths if limit is None else paths[:limit]


class Qwen3VLReranker:
    """Multimodal reranker backed by the official Qwen3-VL-Reranker implementation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Reranker-2B",
        *,
        implementation_repository: str | Path,
        instruction: str = (
            "Given a video question, judge whether the candidate segment contains evidence "
            "needed to answer it."
        ),
        max_frames: int = 16,
        fps: float = 1.0,
        unload_after_score: bool = False,
        model_factory: Any = None,
    ) -> None:
        self.model_name = model_name
        self.implementation_repository = Path(implementation_repository)
        self.instruction = instruction
        self.max_frames = max_frames
        self.fps = fps
        self.unload_after_score = unload_after_score
        self._model_factory = model_factory
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            model_class = self._model_factory or _load_official_class(
                self.implementation_repository,
                "qwen3_vl_reranker",
                "Qwen3VLReranker",
            )
            self._model = model_class(model_name_or_path=self.model_name)
        return self._model

    def score(self, query: str, segments: list[VideoSegment]) -> list[float]:
        if not segments:
            return []
        documents: list[dict[str, Any]] = []
        for segment in segments:
            document: dict[str, Any] = {
                "text": (
                    f"Segment {segment.segment_id}, {segment.start_time:.3f}-"
                    f"{segment.end_time:.3f} seconds\n{segment.searchable_text}"
                )
            }
            images = _existing_images(segment, limit=self.max_frames)
            if images:
                document["video"] = images
            documents.append(document)
        scores = self._load().process(
            {
                "instruction": self.instruction,
                "query": {"text": query},
                "documents": documents,
                "fps": self.fps,
                "max_frames": self.max_frames,
            }
        )
        result = [float(score) for score in scores]
        if self.unload_after_score:
            self.unload()
        return result

    def unload(self) -> None:
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
