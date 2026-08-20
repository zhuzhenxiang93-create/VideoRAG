from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize one immutable formal human-review session.")
    parser.add_argument("--candidates", type=Path, default=Path("data/evaluation/questions.zh.review_queue.v1.jsonl"))
    parser.add_argument("--guide", type=Path, default=Path("docs/ANNOTATION_GUIDE_ZH.md"))
    parser.add_argument("--events", type=Path, default=Path("artifacts/annotations/review_events.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/annotations/review_session_manifest.json"))
    parser.add_argument("--reviewer-id", default="project_owner_zzx")
    args = parser.parse_args()
    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.events.touch(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "formal_human_review_session",
        "reviewer_id": args.reviewer_id,
        "review_timestamp_timezone": "UTC",
        "candidate_file": str(args.candidates),
        "candidate_sha256": sha256(args.candidates),
        "candidate_count": sum(1 for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()),
        "annotation_guide": str(args.guide),
        "annotation_guide_sha256": sha256(args.guide),
        "code_commit": commit,
        "event_log": str(args.events),
        "backup_directory": str(args.events.parent / "backups"),
        "pilot_target": {
            "total": "10-15",
            "audio": 3, "visual": 3, "ocr": 2, "multimodal": 3, "unanswerable": "2-3",
            "requires_reject": True, "requires_reopen": True,
        },
        "formal_metric_eligible": False,
        "eligibility_condition": ">=100 human-reviewed verified questions plus frozen video-disjoint splits",
    }
    if args.manifest.exists():
        previous = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest["created_at"] = previous["created_at"]
        if previous != manifest:
            raise ValueError("formal review session manifest differs; create a new version instead of overwriting")
    else:
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    backup_dir = args.events.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    initial = backup_dir / "review_events.sequence_000000.jsonl"
    if not initial.exists():
        initial.write_bytes(args.events.read_bytes())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
