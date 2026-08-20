from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from video_rag.config import load_config
from video_rag.retrieval import FrameClipVisionRetriever
from video_rag.storage import load_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an independent unique-physical-frame Chinese-CLIP index.")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/frame_index_build.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    segments = load_segments(args.segments)
    retriever = FrameClipVisionRetriever(
        config.models.clip,
        device=args.device,
        index_dir=args.index_dir,
        batch_size=args.batch_size,
        force_rebuild=args.force,
    )
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    started = perf_counter()
    retriever.build(segments, segments_path=args.segments)
    elapsed = perf_counter() - started
    peak = None
    reserved = None
    if torch is not None and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.max_memory_reserved() / 1024**2
    manifest = json.loads((args.index_dir / "vision_frame_zh.manifest.json").read_text(encoding="utf-8"))
    report = {
        "status": "measured",
        "seconds": elapsed,
        "batch_size": args.batch_size,
        "peak_gpu_allocated_mib": peak,
        "peak_gpu_reserved_mib": reserved,
        "manifest": manifest,
        "index_bytes": (args.index_dir / "vision_frame_zh.faiss").stat().st_size,
        "metadata_bytes": (args.index_dir / "vision_frame_zh.json").stat().st_size,
        "old_segment_index_preserved": (args.index_dir / "vision_dense_zh.faiss").exists(),
        "qwen_vl_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
