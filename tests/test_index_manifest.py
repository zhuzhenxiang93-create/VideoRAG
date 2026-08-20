import json

import pytest

from video_rag.index_manifest import SIMILARITY, validate_manifest


def base_manifest(segments_path, model="clip-model"):
    import hashlib
    return {
        "schema_version": 2,
        "segment_count": 1,
        "segments_sha256": hashlib.sha256(segments_path.read_bytes()).hexdigest(),
        "models": {"text_embedding": "text-model", "clip": model},
        "similarity": SIMILARITY,
        "indexes": {},
    }


def test_rejects_wrong_model_before_loading_indexes(tmp_path):
    segments = tmp_path / "segments.jsonl"
    segments.write_text('{"segment_id":"s1"}\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(base_manifest(segments, model="wrong")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="model mismatch"):
        validate_manifest(
            segments_path=segments,
            index_dir=tmp_path,
            text_model="text-model",
            clip_model="clip-model",
        )


def test_rejects_wrong_segments_sha_before_loading_indexes(tmp_path):
    segments = tmp_path / "segments.jsonl"
    segments.write_text('{"segment_id":"s1"}\n', encoding="utf-8")
    manifest = base_manifest(segments)
    manifest["segments_sha256"] = "bad"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_manifest(
            segments_path=segments,
            index_dir=tmp_path,
            text_model="text-model",
            clip_model="clip-model",
        )


def test_rejects_legacy_manifest(tmp_path):
    segments = tmp_path / "segments.jsonl"
    segments.write_text('{"segment_id":"s1"}\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        validate_manifest(
            segments_path=segments,
            index_dir=tmp_path,
            text_model="text-model",
            clip_model="clip-model",
        )
