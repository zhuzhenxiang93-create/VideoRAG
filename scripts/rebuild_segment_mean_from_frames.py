from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from video_rag.config import load_config
from video_rag.index_manifest import file_sha256
from video_rag.retrieval.faiss_dense import normalize_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild E0 segment means from one-time physical-frame encodings.")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--old-index-dir", type=Path, default=None)
    parser.add_argument("--skip-old-comparison", action="store_true")
    parser.add_argument("--output-name", default="vision_segment_mean_unique_zh")
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation/e0_unique_frame_parity.json"))
    args = parser.parse_args()
    import faiss

    config = load_config(args.config)
    frame_index = faiss.read_index(str(args.index_dir / "vision_frame_zh.faiss"))
    frames = json.loads((args.index_dir / "vision_frame_zh.json").read_text(encoding="utf-8"))["frames"]
    if any("raw_embedding_norm" not in item for item in frames):
        raise ValueError("frame metadata lacks raw_embedding_norm; rebuild frame index first")
    normalized = frame_index.reconstruct_n(0, frame_index.ntotal)
    raw_by_segment: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for frame, vector in zip(frames, normalized, strict=True):
        raw = vector * float(frame["raw_embedding_norm"])
        for segment_id in frame["segment_ids"]:
            raw_by_segment[segment_id].append(raw)
    segment_ids = sorted(raw_by_segment)
    vectors = normalize_rows(np.stack([np.mean(raw_by_segment[item], axis=0) for item in segment_ids]))
    output_index = faiss.IndexFlatIP(vectors.shape[1])
    output_index.add(vectors)
    index_path = args.index_dir / f"{args.output_name}.faiss"
    metadata_path = args.index_dir / f"{args.output_name}.json"
    manifest_path = args.index_dir / f"{args.output_name}.manifest.json"
    faiss.write_index(output_index, str(index_path))
    metadata_path.write_text(json.dumps({"segment_ids": segment_ids}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.models.clip,
        "index_granularity": "segment",
        "aggregation": "mean raw physical-frame embeddings, then L2 normalize",
        "source_frame_index": "vision_frame_zh",
        "dimension": int(output_index.d),
        "count": int(output_index.ntotal),
        "segments_sha256": file_sha256(args.segments),
        "index_sha256": file_sha256(index_path),
        "metadata_sha256": file_sha256(metadata_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "status": "measured",
        "manifest": manifest,
    }
    if args.skip_old_comparison:
        report["comparison"] = "skipped because supplemented segments intentionally change E0 vectors"
    else:
        old_dir = args.old_index_dir or args.index_dir
        old_index = faiss.read_index(str(old_dir / "vision_dense_zh.faiss"))
        old_ids = json.loads((old_dir / "vision_dense_zh.json").read_text(encoding="utf-8"))["segment_ids"]
        new_position = {segment_id: index for index, segment_id in enumerate(segment_ids)}
        cosine = []
        max_abs = []
        for old_position, segment_id in enumerate(old_ids):
            old_vector = old_index.reconstruct(old_position)
            new_vector = output_index.reconstruct(new_position[segment_id])
            cosine.append(float(np.dot(old_vector, new_vector)))
            max_abs.append(float(np.max(np.abs(old_vector - new_vector))))
        report.update({
            "comparison": "old per-segment repeated encoding vs unique physical-frame encoding + raw-norm reconstruction",
            "segments_compared": len(cosine),
            "cosine_similarity": {"min": min(cosine), "mean": float(np.mean(cosine)), "max": max(cosine)},
            "max_absolute_difference": {"max": max(max_abs), "mean": float(np.mean(max_abs))},
            "exact_float_equality_expected": False,
            "reason": "GPU batching order can cause small floating-point differences; ranking parity is evaluated separately",
        })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
