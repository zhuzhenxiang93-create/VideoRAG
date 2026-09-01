from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SIMILARITY = "cosine via L2-normalized float32 + IndexFlatIP"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_spec(index_dir: Path, name: str) -> dict[str, Any]:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Index manifest inspection requires faiss") from exc
    index_path = index_dir / f"{name}.faiss"
    metadata_path = index_dir / f"{name}.json"
    if not index_path.exists() or not metadata_path.exists():
        raise ValueError(f"missing index files for {name}")
    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    segment_ids = list(metadata.get("segment_ids", []))
    if index.ntotal != len(segment_ids):
        raise ValueError(
            f"{name} index count {index.ntotal} != metadata count {len(segment_ids)}"
        )
    return {
        "dimension": int(index.d),
        "count": int(index.ntotal),
        "index_sha256": file_sha256(index_path),
        "metadata_sha256": file_sha256(metadata_path),
    }


def build_manifest(
    *, segments_path: Path, index_dir: Path, text_model: str, clip_model: str
) -> dict[str, Any]:
    segment_count = sum(
        1 for line in segments_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    vision_name = "vision_dense_zh" if "chinese-clip" in clip_model.lower() else "vision_dense"
    return {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "segment_count": segment_count,
        "segments_sha256": file_sha256(segments_path),
        "models": {"text_embedding": text_model, "clip": clip_model},
        "similarity": SIMILARITY,
        "indexes": {
            "text_dense": index_spec(index_dir, "text_dense"),
            vision_name: index_spec(index_dir, vision_name),
        },
    }


def validate_manifest(
    *, segments_path: Path, index_dir: Path, text_model: str, clip_model: str
) -> dict[str, Any]:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("index manifest is missing; rebuild or refresh indexes")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported or legacy index manifest; refresh indexes")
    expected = {
        "text_embedding": text_model,
        "clip": clip_model,
    }
    if manifest.get("models") != expected:
        raise ValueError(
            f"index model mismatch: manifest={manifest.get('models')} config={expected}"
        )
    actual_sha = file_sha256(segments_path)
    if manifest.get("segments_sha256") != actual_sha:
        raise ValueError("segments SHA-256 does not match the index manifest")
    actual_count = sum(
        1 for line in segments_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if manifest.get("segment_count") != actual_count:
        raise ValueError("segment count does not match the index manifest")
    if manifest.get("similarity") != SIMILARITY:
        raise ValueError("index similarity definition does not match the runtime")

    for name, expected_spec in manifest.get("indexes", {}).items():
        actual_spec = index_spec(index_dir, name)
        for field in ("dimension", "count", "index_sha256", "metadata_sha256"):
            if expected_spec.get(field) != actual_spec[field]:
                raise ValueError(f"{name} {field} does not match the index manifest")
    required = {"text_dense", "vision_dense_zh" if "chinese-clip" in clip_model.lower() else "vision_dense"}
    if set(manifest.get("indexes", {})) != required:
        raise ValueError("index manifest does not contain exactly the required runtime indexes")
    return manifest


def build_runtime_manifest(
    *,
    segments_path: Path,
    index_dir: Path,
    index_models: dict[str, str],
) -> dict[str, Any]:
    """Build a backend-neutral manifest for an explicit set of runtime indexes."""
    segment_count = sum(
        1 for line in segments_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    return {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "segment_count": segment_count,
        "segments_sha256": file_sha256(segments_path),
        "index_models": index_models,
        "similarity": SIMILARITY,
        "indexes": {name: index_spec(index_dir, name) for name in index_models},
    }


def validate_runtime_manifest(
    *,
    segments_path: Path,
    index_dir: Path,
    index_models: dict[str, str],
) -> dict[str, Any]:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("index manifest is missing; rebuild indexes")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise ValueError("Qwen3-VL runtime requires a schema v3 index manifest; rebuild indexes")
    if manifest.get("index_models") != index_models:
        raise ValueError(
            f"index model mismatch: manifest={manifest.get('index_models')} "
            f"config={index_models}"
        )
    if manifest.get("segments_sha256") != file_sha256(segments_path):
        raise ValueError("segments SHA-256 does not match the index manifest")
    actual_count = sum(
        1 for line in segments_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if manifest.get("segment_count") != actual_count:
        raise ValueError("segment count does not match the index manifest")
    if manifest.get("similarity") != SIMILARITY:
        raise ValueError("index similarity definition does not match the runtime")
    if set(manifest.get("indexes", {})) != set(index_models):
        raise ValueError("index manifest does not contain exactly the required runtime indexes")
    for name, expected_spec in manifest["indexes"].items():
        actual_spec = index_spec(index_dir, name)
        for field in ("dimension", "count", "index_sha256", "metadata_sha256"):
            if expected_spec.get(field) != actual_spec[field]:
                raise ValueError(f"{name} {field} does not match the index manifest")
    return manifest
