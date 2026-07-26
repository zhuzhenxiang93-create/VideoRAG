from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from video_rag.config import load_config
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
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segment_count": len(segments),
        "segments_sha256": hashlib.sha256(args.segments.read_bytes()).hexdigest(),
        "models": {
            "text_embedding": config.models.text_embedding,
            "clip": config.models.clip,
        },
        "similarity": "cosine via L2-normalized float32 + IndexFlatIP",
    }
    (args.index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Indexes saved to {args.index_dir}")


if __name__ == "__main__":
    main()
