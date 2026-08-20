import json

import pytest

from video_rag.evaluation.review_events import load_review_events


def event(event_id, sequence, action, previous, sha="sha"):
    return {
        "event_id": event_id, "sequence": sequence, "previous_event_id": previous,
        "question_id": "q", "action": action, "candidate_file_sha256": sha,
    }


def write(path, rows):
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


@pytest.mark.parametrize(
    "actions,expected",
    [
        (["accept", "reject"], "reject"),
        (["reject", "accept"], "accept"),
        (["accept", "accept"], "accept"),
    ],
)
def test_latest_file_order_decision_wins(actions, expected, tmp_path):
    rows, previous = [], None
    for index, action in enumerate(actions, 1):
        current = f"e{index}"
        rows.append(event(current, index, action, previous))
        previous = current
    path = tmp_path / "events.jsonl"
    write(path, rows)
    _, latest = load_review_events(path, candidate_sha256="sha", known_question_ids={"q"})
    assert latest["q"]["action"] == expected


def test_duplicate_event_id_and_non_monotonic_sequence_fail_closed(tmp_path):
    path = tmp_path / "events.jsonl"
    write(path, [event("same", 1, "accept", None), event("same", 2, "reject", "same")])
    with pytest.raises(ValueError, match="duplicate"):
        load_review_events(path, candidate_sha256="sha", known_question_ids={"q"})
    write(path, [event("e1", 2, "accept", None)])
    with pytest.raises(ValueError, match="non-monotonic"):
        load_review_events(path, candidate_sha256="sha", known_question_ids={"q"})


def test_sha_mismatch_and_corrupted_last_line_fail_closed(tmp_path):
    path = tmp_path / "events.jsonl"
    write(path, [event("e1", 1, "accept", None, sha="wrong")])
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_review_events(path, candidate_sha256="sha", known_question_ids={"q"})
    path.write_text(json.dumps(event("e1", 1, "accept", None)) + "\n{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted"):
        load_review_events(path, candidate_sha256="sha", known_question_ids={"q"})
