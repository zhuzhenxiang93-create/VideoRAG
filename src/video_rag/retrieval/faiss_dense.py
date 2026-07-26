from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any

import numpy as np

from video_rag.schemas import SearchHit, VideoSegment


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot index or search a zero vector")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class FaissDenseRetriever(ABC):
    """Cosine search implemented as inner product over normalized float32 vectors."""

    name = "dense"

    def __init__(self, index_dir: str | Path | None = None, *, force_rebuild: bool = False) -> None:
        self.index_dir = Path(index_dir) if index_dir else None
        self.force_rebuild = force_rebuild
        self._index: Any = None
        self._segment_ids: list[str] = []

    @abstractmethod
    def encode_documents(self, segments: list[VideoSegment]) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def build(self, segments: list[VideoSegment]) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Dense retrieval requires faiss-cpu or faiss-gpu") from exc
        index_path = self.index_dir / f"{self.name}.faiss" if self.index_dir else None
        metadata_path = self.index_dir / f"{self.name}.json" if self.index_dir else None
        if (
            not self.force_rebuild
            and index_path
            and metadata_path
            and index_path.exists()
            and metadata_path.exists()
        ):
            self.load(self.index_dir)
            known_ids = {segment.segment_id for segment in segments}
            if not set(self._segment_ids).issubset(known_ids):
                raise ValueError(f"{self.name} index contains unknown segment IDs")
            return

        vectors, segment_ids = self.encode_documents(segments)
        vectors = normalize_rows(vectors)
        if len(vectors) != len(segment_ids):
            raise ValueError("Encoder returned mismatched vectors and segment IDs")
        if not segment_ids:
            raise ValueError(f"{self.name} produced no indexable documents")
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._segment_ids = segment_ids
        if self.index_dir:
            self.save(self.index_dir)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if self._index is None:
            if self.index_dir and (self.index_dir / f"{self.name}.faiss").exists():
                self.load(self.index_dir)
            else:
                raise RuntimeError(f"{self.name} index has not been built")
        if top_k <= 0:
            return []
        vector = normalize_rows(self.encode_query(query))
        count = min(top_k, len(self._segment_ids))
        scores, indices = self._index.search(vector, count)
        return [
            SearchHit(
                segment_id=self._segment_ids[int(index)],
                score=float(score),
                source=self.name,
                rank=rank,
            )
            for rank, (index, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1)
            if 0 <= int(index) < len(self._segment_ids)
        ]

    def save(self, directory: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("Cannot save an empty index")
        import faiss

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(destination / f"{self.name}.faiss"))
        (destination / f"{self.name}.json").write_text(
            json.dumps({"segment_ids": self._segment_ids}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, directory: str | Path) -> None:
        import faiss

        source = Path(directory)
        metadata = json.loads((source / f"{self.name}.json").read_text(encoding="utf-8"))
        self._index = faiss.read_index(str(source / f"{self.name}.faiss"))
        self._segment_ids = list(metadata["segment_ids"])
        if self._index.ntotal != len(self._segment_ids):
            raise ValueError(f"{self.name} index and metadata are inconsistent")


class QwenTextRetriever(FaissDenseRetriever):
    name = "text_dense"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        device: str = "cuda",
        index_dir: str | Path | None = None,
        force_rebuild: bool = False,
    ) -> None:
        super().__init__(index_dir, force_rebuild=force_rebuild)
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Qwen embedding requires sentence-transformers") from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_documents(self, segments: list[VideoSegment]) -> tuple[np.ndarray, list[str]]:
        indexable = [segment for segment in segments if segment.searchable_text.strip()]
        vectors = self._load().encode(
            [segment.searchable_text for segment in indexable],
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vectors, dtype=np.float32), [item.segment_id for item in indexable]

    def encode_query(self, query: str) -> np.ndarray:
        vector = self._load().encode(
            [query],
            prompt_name="query",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vector, dtype=np.float32)


class ClipVisionRetriever(FaissDenseRetriever):
    name = "vision_dense"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        *,
        device: str = "cuda",
        index_dir: str | Path | None = None,
        force_rebuild: bool = False,
    ) -> None:
        super().__init__(index_dir, force_rebuild=force_rebuild)
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def _load(self):
        if self._model is None:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as exc:
                raise RuntimeError("CLIP retrieval requires torch and transformers") from exc
            actual_device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
            self._model = CLIPModel.from_pretrained(self.model_name).eval().to(actual_device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self.device = actual_device
        return self._model, self._processor

    def encode_documents(self, segments: list[VideoSegment]) -> tuple[np.ndarray, list[str]]:
        import torch
        from PIL import Image

        model, processor = self._load()
        vectors: list[np.ndarray] = []
        identifiers: list[str] = []
        with torch.inference_mode():
            for segment in segments:
                paths = [frame.path for frame in segment.keyframes if Path(frame.path).exists()]
                if not paths:
                    continue
                images = [Image.open(path).convert("RGB") for path in paths]
                try:
                    inputs = processor(images=images, return_tensors="pt", padding=True).to(self.device)
                    features = model.get_image_features(**inputs)
                    pooled = features.float().mean(dim=0, keepdim=True)
                    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
                    vectors.append(pooled.cpu().numpy()[0])
                    identifiers.append(segment.segment_id)
                finally:
                    for image in images:
                        image.close()
        return np.asarray(vectors, dtype=np.float32), identifiers

    def encode_query(self, query: str) -> np.ndarray:
        import torch

        model, processor = self._load()
        with torch.inference_mode():
            inputs = processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            vector = model.get_text_features(**inputs).float()
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector.cpu().numpy().astype(np.float32)
