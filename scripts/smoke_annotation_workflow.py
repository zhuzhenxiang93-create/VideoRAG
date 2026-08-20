from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

from scripts.annotation_server import create_app
from video_rag.evaluation.review_events import load_review_events


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated end-to-end annotation smoke test; never writes formal reviews.")
    parser.add_argument("--candidates", type=Path, default=Path("data/evaluation/questions.zh.review_queue.v1.jsonl"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.supplement_a.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/annotation_smoke_latest.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidate = next(item for item in rows if item["question_type"] == "audio" and not item.get("review_flags"))
    run_dir = Path("artifacts/smoke_annotation") / str(uuid.uuid4())
    run_dir.mkdir(parents=True)
    candidate_path = run_dir / "candidate.jsonl"
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8")
    events_path = run_dir / "events.jsonl"
    export_path = run_dir / "verified.jsonl"
    export_report = run_dir / "export_report.json"

    def app_client():
        return create_app(
            candidate_path, args.segments, events_path, "automated_smoke_only_not_human_annotation",
            allowed_video_root=Path("data/raw"), allowed_frame_root=Path("artifacts"),
        ).test_client()

    client = app_client()
    next_payload = client.get("/api/next").get_json()
    assert next_payload["item"]["question"]["question_id"] == candidate["question_id"]
    media = client.get(next_payload["item"]["video_url"], headers={"Range": "bytes=0-1023"})
    assert media.status_code in {200, 206}
    if next_payload["item"]["frame_urls"]:
        assert client.get(next_payload["item"]["frame_urls"][0]).status_code == 200

    edited = json.loads(json.dumps(candidate, ensure_ascii=False))
    edited["modality_evidence"]["asr"]["human_verified"] = True
    behavior = {
        "video_played": True,
        "playback_ranges": [[float(candidate["evidence_start"]), float(candidate["evidence_end"])]],
        "review_duration_seconds": 30.0,
        "smoke_simulated": True,
    }
    accept = client.post("/api/review", json={
        "question_id": candidate["question_id"], "base_event_id": None, "action": "accept",
        "reason": "automated isolated smoke", "edited": edited, "review_behavior": behavior,
    })
    assert accept.status_code == 200, accept.get_json()
    accept_id = accept.get_json()["event_id"]

    def export() -> dict:
        subprocess.run([
            sys.executable, "scripts/export_verified_questions.py", "--candidates", str(candidate_path),
            "--events", str(events_path), "--segments", str(args.segments), "--output", str(export_path),
            "--report", str(export_report),
        ], check=True, capture_output=True, text=True)
        return json.loads(export_report.read_text(encoding="utf-8"))

    first_export = export()
    assert first_export["accepted"] == 1
    first_sha = first_export["output_sha256"]
    assert export()["output_sha256"] == first_sha

    # Recreate the Flask app to prove event-chain recovery after service restart.
    client = app_client()
    reopen = client.post("/api/reopen", json={
        "question_id": candidate["question_id"], "base_event_id": accept_id, "reason": "smoke reopen",
    })
    assert reopen.status_code == 200
    reopen_id = reopen.get_json()["event_id"]
    stale = client.post("/api/review", json={
        "question_id": candidate["question_id"], "base_event_id": accept_id, "action": "reject",
        "reason": "stale tab", "edited": candidate,
    })
    assert stale.status_code == 409
    reject = client.post("/api/review", json={
        "question_id": candidate["question_id"], "base_event_id": reopen_id, "action": "reject",
        "reason": "smoke reject", "edited": candidate,
    })
    assert reject.status_code == 200
    reject_id = reject.get_json()["event_id"]
    assert export()["accepted"] == 0

    edited["answer_aliases"] = list(dict.fromkeys([*edited["answer_aliases"], edited["answer"]]))
    final_accept = client.post("/api/review", json={
        "question_id": candidate["question_id"], "base_event_id": reject_id, "action": "accept",
        "reason": "smoke final accept", "edited": edited, "review_behavior": behavior,
    })
    assert final_accept.status_code == 200, final_accept.get_json()
    final_export = export()
    assert final_export["accepted"] == 1

    event_rows, latest = load_review_events(
        events_path, candidate_sha256=sha256(candidate_path), known_question_ids={candidate["question_id"]}
    )
    assert latest[candidate["question_id"]]["event_id"] == final_accept.get_json()["event_id"]
    assert latest[candidate["question_id"]]["annotation_after"]["review_event_id"] == final_accept.get_json()["event_id"]
    report = {
        "status": "passed", "formal_metric_eligible": False,
        "warning": "automated isolated smoke events are not human annotations",
        "run_directory": str(run_dir), "candidate_sha256": sha256(candidate_path),
        "event_sha256": sha256(events_path), "event_count": len(event_rows),
        "sequence": [item["action"] for item in event_rows],
        "checks": {
            "media_range": True, "frame_if_present": True, "accept_export": True,
            "stable_export_sha": True, "restart_recovery": True, "reopen_reject_removes_export": True,
            "stale_revision_409": True, "reject_then_accept_exports_latest": True,
            "cross_reference_validation": True,
        },
        "final_export_sha256": final_export["output_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
