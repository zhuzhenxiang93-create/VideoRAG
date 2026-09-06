"""Extract only the first five selected tutorials, retaining the original archive."""

import json
import zipfile
from pathlib import Path

selected = ["14643", "14644", "14645", "14646", "14647"]
root = Path("data/tutorialvqa/videos")
root.mkdir(parents=True, exist_ok=True)
records = []
with zipfile.ZipFile("data/tutorialvqa/original/videos.zip") as archive:
    for vid in selected:
        name = f"videos/{vid}.webm"
        info = archive.getinfo(name)
        destination = root / f"{vid}.webm"
        if not destination.exists():
            destination.write_bytes(archive.read(name))
        import zlib

        data = destination.read_bytes()
        assert len(data) == info.file_size and zlib.crc32(data) == info.CRC
        records.append(
            {
                "video_id": vid,
                "path": str(destination.resolve()),
                "bytes": len(data),
                "crc32": info.CRC,
            }
        )
Path("artifacts/tutorialvqa/selected-files.json").write_text(json.dumps(records, indent=2))
print("Verified", len(records), "videos")
