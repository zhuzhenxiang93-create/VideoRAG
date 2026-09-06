"""Fetch official raw files on the school server. No QA text enters an index.

Uses standard HTTPS_PROXY/HTTP_PROXY if supplied by the school environment.
Keeps original splits unchanged; converter must be written after schema inspection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

REPOSITORY = "https://api.github.com/repos/acolas1/TutorialVQAData"
MIRROR = "https://archive.org/metadata/videos_202604"


def fetch(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Preserving {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "VideoRAG-research"})
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.rename(destination)
    print(f"Saved {destination} ({destination.stat().st_size} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/tutorialvqa/original"))
    parser.add_argument(
        "--download-videos", action="store_true", help="Download official videos.zip (~3.2 GB)"
    )
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    fetch(REPOSITORY, root / "repository-metadata.json")
    branch = json.loads((root / "repository-metadata.json").read_text())["default_branch"]
    fetch(REPOSITORY + f"/git/trees/{branch}?recursive=1", root / "repository-tree.json")
    tree = json.loads((root / "repository-tree.json").read_text())["tree"]
    required = {"train.json", "dev.json", "test.json", "videos.json"}
    found = set()
    for item in tree:
        path = item["path"]
        name = Path(path).name
        if item["type"] != "blob" or not (
            name in required or name.lower().startswith(("readme", "license"))
        ):
            continue
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Unsafe repository path")
        fetch(
            f"https://raw.githubusercontent.com/acolas1/TutorialVQAData/{branch}/{path}",
            root / "repository" / path,
        )
        found.add(name)
    if required - found:
        raise RuntimeError(f"Missing official JSON files: {required - found}")
    fetch(MIRROR, root / "archive-metadata.json")
    metadata = json.loads((root / "archive-metadata.json").read_text())
    if args.download_videos:
        entry = next(f for f in metadata["files"] if f["name"] == "videos.zip")
        fetch("https://archive.org/download/videos_202604/videos.zip", root / "videos.zip")
        if (root / "videos.zip").stat().st_size != int(entry["size"]):
            raise RuntimeError("Downloaded video archive size differs from mirror metadata")
    print(
        "Original files preserved. Inspect actual schema and evidence boundaries before conversion."
    )


if __name__ == "__main__":
    main()
