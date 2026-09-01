from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_rag.config import load_config
from video_rag.index_manifest import build_manifest, build_runtime_manifest
from video_rag.retrieval import (
    ClipVisionRetriever,
    Qwen3VLEmbeddingRetriever,
    QwenTextRetriever,
)
from video_rag.storage import load_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persistent FAISS indexes.")
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--index-dir", default=Path("artifacts/indexes"), type=Path)
    parser.add_argument("--config", default=Path("config.toml"), type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    segments = load_segments(args.segments)
    text_retriever = QwenTextRetriever(
        config.models.text_embedding,
        device=args.device,
        index_dir=args.index_dir,
        force_rebuild=True,
    )
    if config.retrieval.vision_backend == "qwen3_vl":
        if not config.models.qwen3_vl_repository:
            raise ValueError("models.qwen3_vl_repository is required for Qwen3-VL indexing")
        vision_retriever = Qwen3VLEmbeddingRetriever(
            config.models.qwen3_vl_embedding,
            implementation_repository=config.models.qwen3_vl_repository,
            index_dir=args.index_dir,
            force_rebuild=True,
            max_frames=config.generation.max_frames,
            fps=config.generation.video_fps,
        )
    else:
        vision_retriever = ClipVisionRetriever(
            config.models.clip,
            device=args.device,
            index_dir=args.index_dir,
            force_rebuild=True,
        )
    retrievers = [text_retriever, vision_retriever]
    for retriever in retrievers:
        print(f"Building {retriever.name}...")
        retriever.build(segments)
    if config.retrieval.vision_backend == "qwen3_vl":
        manifest = build_runtime_manifest(
            segments_path=args.segments,
            index_dir=args.index_dir,
            index_models={
                text_retriever.name: config.models.text_embedding,
                vision_retriever.name: config.models.qwen3_vl_embedding,
            },
        )
    else:
        manifest = build_manifest(
            segments_path=args.segments,
            index_dir=args.index_dir,
            text_model=config.models.text_embedding,
            clip_model=config.models.clip,
        )
    (args.index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Indexes saved to {args.index_dir}")


if __name__ == "__main__":
    main()
