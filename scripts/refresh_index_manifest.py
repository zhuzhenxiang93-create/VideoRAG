from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_rag.config import load_config
from video_rag.index_manifest import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh a validated manifest for existing indexes.")
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--index-dir", default=Path("artifacts/indexes"), type=Path)
    parser.add_argument("--config", default=Path("config.toml"), type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = build_manifest(
        segments_path=args.segments,
        index_dir=args.index_dir,
        text_model=config.models.text_embedding,
        clip_model=config.models.clip,
    )
    destination = args.index_dir / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
