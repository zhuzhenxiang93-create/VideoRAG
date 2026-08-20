from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from video_rag.index_manifest import file_sha256
from video_rag.retrieval.frame_faiss import collect_physical_frames
from video_rag.storage import load_segments


def copy_verified(source: Path, destination: Path) -> None:
    if destination.exists():
        if file_sha256(source) != file_sha256(destination):
            raise ValueError(f"frozen baseline differs from current source: {destination}")
        return
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze immutable E0 visual baseline inputs and indexes.")
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--destination", type=Path, default=Path("artifacts/baselines/p1c_e0"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    sources = {
        "config.toml": args.config,
        "segments.jsonl": args.segments,
        "vision_segment_mean_zh.faiss": args.index_dir / "vision_dense_zh.faiss",
        "vision_segment_mean_zh.json": args.index_dir / "vision_dense_zh.json",
        "runtime_index_manifest.json": args.index_dir / "manifest.json",
    }
    for name, source in sources.items():
        copy_verified(source, args.destination / name)

    segments = load_segments(args.segments)
    frames = collect_physical_frames(segments)
    frame_list_path = args.destination / "physical_frames.json"
    frame_list_content = json.dumps({"frames": frames}, ensure_ascii=False, indent=2) + "\n"
    if frame_list_path.exists() and frame_list_path.read_text(encoding="utf-8") != frame_list_content:
        raise ValueError("frozen physical frame list differs from current data")
    frame_list_path.write_text(frame_list_content, encoding="utf-8")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "immutable E0: original scene keyframes + segment mean Chinese-CLIP",
        "git_commit": commit,
        "files": {
            name: {"bytes": (args.destination / name).stat().st_size, "sha256": file_sha256(args.destination / name)}
            for name in [*sources, "physical_frames.json"]
        },
        "physical_frame_count": len(frames),
        "membership_count": sum(len(item["segment_ids"]) for item in frames),
        "formal_metric_eligible": False,
        "metric_note": "the current 20-question set is generated_candidate, not human verified",
    }
    manifest_path = args.destination / "baseline_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["created_at"] = previous["created_at"]
        if previous != manifest:
            raise ValueError("existing baseline manifest differs; baseline is immutable")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
