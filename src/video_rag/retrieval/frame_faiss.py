from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from video_rag.index_manifest import file_sha256
from video_rag.retrieval.faiss_dense import normalize_rows, unwrap_model_features
from video_rag.schemas import SearchHit, VideoSegment


FRAME_INDEX_SIMILARITY = "cosine via L2-normalized float32 + IndexFlatIP"


def collect_physical_frames(segments: Iterable[VideoSegment]) -> list[dict[str, Any]]:
    """Create one record per physical image and preserve all segment memberships."""
    records: dict[str, dict[str, Any]] = {}
    for segment in segments:
        for frame in segment.keyframes:
            normalized_path = Path(frame.path).as_posix()
            physical_key = f"{segment.video_id}\0{frame.timestamp:.3f}\0{normalized_path}"
            frame_id = hashlib.sha256(physical_key.encode("utf-8")).hexdigest()[:24]
            record = records.setdefault(
                normalized_path,
                {
                    "frame_id": frame_id,
                    "frame_path": normalized_path,
                    "video_id": segment.video_id,
                    "timestamp": float(frame.timestamp),
                    "segment_ids": [],
                },
            )
            if record["video_id"] != segment.video_id or record["timestamp"] != float(frame.timestamp):
                raise ValueError(f"inconsistent metadata for physical frame {normalized_path}")
            if segment.segment_id not in record["segment_ids"]:
                record["segment_ids"].append(segment.segment_id)
    return sorted(records.values(), key=lambda item: (item["video_id"], item["timestamp"], item["frame_path"]))


def aggregate_frame_hits(
    frame_scores: Iterable[tuple[dict[str, Any], float]], aggregation: str
) -> list[tuple[str, float]]:
    """Expand physical-frame scores to segments and aggregate without duplicate voting."""
    if aggregation not in {"max", "top2_mean"}:
        raise ValueError("aggregation must be 'max' or 'top2_mean'")
    scores_by_segment: defaultdict[str, list[float]] = defaultdict(list)
    for frame, score in frame_scores:
        for segment_id in frame["segment_ids"]:
            scores_by_segment[segment_id].append(float(score))
    aggregated = []
    for segment_id, scores in scores_by_segment.items():
        ordered = sorted(scores, reverse=True)
        score = ordered[0] if aggregation == "max" else sum(ordered[:2]) / min(2, len(ordered))
        aggregated.append((segment_id, score))
    return sorted(aggregated, key=lambda item: (-item[1], item[0]))


class FrameClipVisionRetriever:
    """Chinese-CLIP index keyed by unique physical frames, aggregated at query time."""

    index_name = "vision_frame_zh"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        index_dir: str | Path = "artifacts/indexes",
        aggregation: str = "max",
        frame_candidate_k: int = 200,
        batch_size: int = 16,
        force_rebuild: bool = False,
    ) -> None:
        if aggregation not in {"max", "top2_mean"}:
            raise ValueError("aggregation must be 'max' or 'top2_mean'")
        if frame_candidate_k < 1 or batch_size < 1:
            raise ValueError("frame_candidate_k and batch_size must be positive")
        self.model_name = model_name
        self.device = device
        self.index_dir = Path(index_dir)
        self.aggregation = aggregation
        self.frame_candidate_k = frame_candidate_k
        self.batch_size = batch_size
        self.force_rebuild = force_rebuild
        self._index: Any = None
        self._frames: list[dict[str, Any]] = []
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return f"{self.index_name}_{self.aggregation}"

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor

            actual = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
            self._model = ChineseCLIPModel.from_pretrained(self.model_name).eval().to(actual)
            self._processor = ChineseCLIPProcessor.from_pretrained(self.model_name)
            self.device = actual
        return self._model, self._processor

    def _encode_images(self, frames: list[dict[str, Any]]) -> np.ndarray:
        import torch
        from PIL import Image

        model, processor = self._load_model()
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(frames), self.batch_size):
                batch = frames[offset : offset + self.batch_size]
                images = [Image.open(item["frame_path"]).convert("RGB") for item in batch]
                try:
                    inputs = processor(images=images, return_tensors="pt", padding=True).to(self.device)
                    values = unwrap_model_features(model.get_image_features(**inputs)).float()
                    batches.append(values.cpu().numpy())
                finally:
                    for image in images:
                        image.close()
        raw = np.asarray(np.concatenate(batches, axis=0), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1)
        if np.any(norms == 0):
            raise ValueError("Chinese-CLIP produced a zero image vector")
        for frame, norm in zip(frames, norms, strict=True):
            # Retaining the raw norm lets E0 reconstruct mean(raw features)
            # exactly while the frame index itself stores normalized vectors.
            frame["raw_embedding_norm"] = float(norm)
        return normalize_rows(raw)

    def _encode_query(self, query: str) -> np.ndarray:
        import torch

        model, processor = self._load_model()
        with torch.inference_mode():
            inputs = processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            values = unwrap_model_features(model.get_text_features(**inputs)).float().cpu().numpy()
        return normalize_rows(values)

    def build(self, segments: list[VideoSegment], *, segments_path: Path | None = None) -> None:
        if not self.force_rebuild and self._paths_exist():
            self.load(segments_path=segments_path)
            return
        import faiss

        frames = collect_physical_frames(segments)
        missing = [item["frame_path"] for item in frames if not Path(item["frame_path"]).is_file()]
        if missing:
            raise ValueError(f"{len(missing)} frame files are missing; first={missing[0]}")
        vectors = self._encode_images(frames)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._frames = frames
        self.save(segments_path=segments_path)

    def _paths_exist(self) -> bool:
        return all(
            (self.index_dir / name).exists()
            for name in (f"{self.index_name}.faiss", f"{self.index_name}.json", f"{self.index_name}.manifest.json")
        )

    def save(self, *, segments_path: Path | None = None) -> None:
        if self._index is None:
            raise RuntimeError("cannot save an empty frame index")
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / f"{self.index_name}.faiss"
        metadata_path = self.index_dir / f"{self.index_name}.json"
        manifest_path = self.index_dir / f"{self.index_name}.manifest.json"
        faiss.write_index(self._index, str(index_path))
        metadata_path.write_text(json.dumps({"frames": self._frames}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model_name,
            "similarity": FRAME_INDEX_SIMILARITY,
            "index_granularity": "physical_frame",
            "supported_segment_aggregations": ["max", "top2_mean"],
            "dimension": int(self._index.d),
            "physical_frame_count": int(self._index.ntotal),
            "segment_membership_count": sum(len(item["segment_ids"]) for item in self._frames),
            "index_sha256": file_sha256(index_path),
            "metadata_sha256": file_sha256(metadata_path),
            "frame_list_sha256": hashlib.sha256(
                json.dumps(self._frames, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "segments_sha256": file_sha256(segments_path) if segments_path else None,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def load(self, *, segments_path: Path | None = None) -> None:
        import faiss

        index_path = self.index_dir / f"{self.index_name}.faiss"
        metadata_path = self.index_dir / f"{self.index_name}.json"
        manifest_path = self.index_dir / f"{self.index_name}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("model") != self.model_name:
            raise ValueError("frame index manifest schema/model mismatch")
        if manifest.get("similarity") != FRAME_INDEX_SIMILARITY:
            raise ValueError("frame index similarity mismatch")
        if manifest.get("index_granularity") != "physical_frame":
            raise ValueError("frame index granularity mismatch")
        if manifest.get("index_sha256") != file_sha256(index_path) or manifest.get("metadata_sha256") != file_sha256(metadata_path):
            raise ValueError("frame index file hash mismatch")
        if segments_path and manifest.get("segments_sha256") != file_sha256(segments_path):
            raise ValueError("frame index segments hash mismatch")
        self._index = faiss.read_index(str(index_path))
        self._frames = json.loads(metadata_path.read_text(encoding="utf-8"))["frames"]
        actual_frame_list_sha = hashlib.sha256(
            json.dumps(self._frames, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if manifest.get("frame_list_sha256") != actual_frame_list_sha:
            raise ValueError("frame list hash mismatch")
        memberships = sum(len(item["segment_ids"]) for item in self._frames)
        if self._index.ntotal != len(self._frames) or manifest.get("physical_frame_count") != len(self._frames):
            raise ValueError("frame index count does not match metadata")
        if self._index.d != manifest.get("dimension") or memberships != manifest.get("segment_membership_count"):
            raise ValueError("frame index dimension or membership count mismatch")

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            return []
        if self._index is None:
            self.load()
        frame_k = min(max(top_k, self.frame_candidate_k), len(self._frames))
        scores, indices = self._index.search(self._encode_query(query), frame_k)
        frame_hits = [
            (self._frames[int(index)], float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if 0 <= int(index) < len(self._frames)
        ]
        best_frame_by_segment: dict[str, tuple[float, dict[str, Any]]] = {}
        for frame, score in frame_hits:
            for segment_id in frame["segment_ids"]:
                current = best_frame_by_segment.get(segment_id)
                if current is None or score > current[0] or (
                    score == current[0] and frame["frame_id"] < current[1]["frame_id"]
                ):
                    best_frame_by_segment[segment_id] = (score, frame)
        ranked = aggregate_frame_hits(frame_hits, self.aggregation)[:top_k]
        return [
            SearchHit(
                segment_id=segment_id,
                score=score,
                source=self.name,
                rank=rank,
                metadata={
                    "best_frame_id": best_frame_by_segment[segment_id][1]["frame_id"],
                    "best_frame_timestamp": best_frame_by_segment[segment_id][1]["timestamp"],
                    "best_frame_path": best_frame_by_segment[segment_id][1]["frame_path"],
                    "frame_candidate_k": frame_k,
                    "aggregation": self.aggregation,
                },
            )
            for rank, (segment_id, score) in enumerate(ranked, start=1)
        ]
