from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_rag.config import load_config
from video_rag.index_manifest import build_manifest
from video_rag.retrieval import ClipVisionRetriever, QwenTextRetriever
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
    retrievers = [
        QwenTextRetriever(
            config.models.text_embedding,
            device=args.device,
            index_dir=args.index_dir,
            force_rebuild=True,
        ),
        ClipVisionRetriever(
            config.models.clip,
            device=args.device,
            index_dir=args.index_dir,
            force_rebuild=True,
        ),
    ]
    for retriever in retrievers:
        print(f"Building {retriever.name}...")
        retriever.build(segments)
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
