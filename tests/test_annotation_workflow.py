import json

from scripts.annotation_server import create_app


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")


def fixture(tmp_path):
    segment = {
        "segment_id": "video_0000", "video_id": "video", "source_path": str(tmp_path / "video.mp4"),
        "start_time": 0.0, "end_time": 20.0, "transcript": "播报内容", "visual_caption": "画面内容", "keyframes": [],
    }
    question = {
        "question_id": "q1", "question": "谁在播报？", "answer": "记者", "answer_aliases": ["记者"],
        "question_type": "audio", "video_id": "video", "relevant_segment_ids": ["video_0000"],
        "evidence_start": 1.0, "evidence_end": 4.0, "answerable": True,
        "verification_status": "generated_candidate", "annotation_source": "model_generated_pending_review",
    }
    candidates, segments, events = tmp_path / "candidates.jsonl", tmp_path / "segments.jsonl", tmp_path / "events.jsonl"
    write_jsonl(candidates, [question])
    write_jsonl(segments, [segment])
    return candidates, segments, events, question


def test_accept_appends_auditable_human_review_event(tmp_path):
    candidates, segments, events, question = fixture(tmp_path)
    client = create_app(candidates, segments, events, "owner").test_client()
    response = client.post("/api/review", json={
        "question_id": "q1", "action": "accept", "reason": "played evidence", "edited": question,
    })
    assert response.status_code == 200
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["annotation_after"]["verification_status"] == "verified"
    assert rows[0]["annotation_after"]["review_event_id"] == rows[0]["event_id"]
    assert rows[0]["reviewer_id"] == "owner"


def test_reject_requires_reason_and_never_creates_verified_annotation(tmp_path):
    candidates, segments, events, question = fixture(tmp_path)
    client = create_app(candidates, segments, events, "owner").test_client()
    assert client.post("/api/review", json={
        "question_id": "q1", "action": "reject", "reason": "", "edited": question,
    }).status_code == 400
    response = client.post("/api/review", json={
        "question_id": "q1", "action": "reject", "reason": "ambiguous_answer", "edited": question,
    })
    assert response.status_code == 200
    row = json.loads(events.read_text(encoding="utf-8").strip())
    assert row["annotation_after"] is None
