from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_review_events(
    path: Path, *, candidate_sha256: str, known_question_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate an append-only event log and return latest decision by file order."""
    if not path.exists():
        return [], {}
    events = []
    event_ids = set()
    latest: dict[str, dict[str, Any]] = {}
    previous_sequence = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"review log line {line_number} is corrupted: {error.msg}") from error
        event_id = str(item.get("event_id", ""))
        if not event_id or event_id in event_ids:
            raise ValueError(f"review log line {line_number} has duplicate/empty event_id")
        event_ids.add(event_id)
        sequence = item.get("sequence")
        if not isinstance(sequence, int) or sequence != previous_sequence + 1:
            raise ValueError(f"review log line {line_number} has non-monotonic sequence")
        previous_sequence = sequence
        question_id = item.get("question_id")
        if question_id not in known_question_ids:
            raise ValueError(f"review log line {line_number} references unknown question")
        if item.get("candidate_file_sha256") != candidate_sha256:
            raise ValueError(f"review log line {line_number} candidate SHA mismatch")
        previous = latest.get(question_id)
        expected_previous = previous["event_id"] if previous else None
        if item.get("previous_event_id") != expected_previous:
            raise ValueError(f"review log line {line_number} breaks question revision chain")
        if item.get("action") not in {"accept", "reject", "reopen"}:
            raise ValueError(f"review log line {line_number} has invalid action")
        latest[question_id] = item
        events.append(item)
    return events, latest
