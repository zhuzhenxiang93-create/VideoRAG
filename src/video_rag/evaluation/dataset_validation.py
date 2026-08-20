from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


ALLOWED_TYPES = {"audio", "visual", "ocr", "multimodal", "unknown"}
ALLOWED_STATUSES = {"verified", "generated_candidate"}
REQUIRED_FIELDS = {
    "question_id",
    "question",
    "answer",
    "answer_aliases",
    "question_type",
    "video_id",
    "relevant_segment_ids",
    "evidence_start",
    "evidence_end",
    "answerable",
    "verification_status",
    "annotation_source",
}


@dataclass(frozen=True)
class ValidationReport:
    question_count: int
    verified_count: int
    candidate_count: int
    type_counts: dict[str, int]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "question_count": self.question_count,
            "verified_count": self.verified_count,
            "candidate_count": self.candidate_count,
            "type_counts": self.type_counts,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        record["_line_number"] = line_number
        records.append(record)
    return records


def validate_questions(
    questions: list[dict[str, Any]], segments: list[dict[str, Any]]
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    ids: set[str] = set()
    segment_by_id = {item["segment_id"]: item for item in segments}
    status_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for position, item in enumerate(questions, 1):
        line = item.get("_line_number", position)
        missing = sorted(REQUIRED_FIELDS - item.keys())
        if missing:
            errors.append(f"line {line}: missing fields {missing}")
            continue
        question_id = str(item["question_id"]).strip()
        if not question_id or question_id in ids:
            errors.append(f"line {line}: duplicate or empty question_id {question_id!r}")
        ids.add(question_id)

        question_type = item["question_type"]
        type_counts[question_type] += 1
        if question_type not in ALLOWED_TYPES:
            errors.append(f"line {line}: unsupported question_type {question_type!r}")

        status = item["verification_status"]
        status_counts[status] += 1
        if status not in ALLOWED_STATUSES:
            errors.append(f"line {line}: unsupported verification_status {status!r}")
        if status == "verified" and not str(item["annotation_source"]).strip():
            errors.append(f"line {line}: verified item requires annotation_source")

        answerable = item["answerable"]
        if not isinstance(answerable, bool):
            errors.append(f"line {line}: answerable must be boolean")
        if answerable and not str(item["answer"]).strip():
            errors.append(f"line {line}: answerable item requires answer")
        if not answerable and item["relevant_segment_ids"]:
            errors.append(f"line {line}: unanswerable item must not contain relevant segments")

        relevant = item["relevant_segment_ids"]
        if answerable and not relevant:
            errors.append(f"line {line}: answerable item requires relevant segments")
        for segment_id in relevant:
            segment = segment_by_id.get(segment_id)
            if segment is None:
                errors.append(f"line {line}: unknown segment {segment_id!r}")
                continue
            if segment["video_id"] != item["video_id"]:
                errors.append(f"line {line}: segment {segment_id!r} belongs to another video")

        start, end = item["evidence_start"], item["evidence_end"]
        if answerable and not (
            isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and 0 <= start < end
        ):
            errors.append(f"line {line}: invalid evidence interval")
        if not answerable and (start is not None or end is not None):
            errors.append(f"line {line}: unanswerable evidence interval must be null")

    if type_counts and max(type_counts.values()) / sum(type_counts.values()) > 0.70:
        warnings.append("question types are severely imbalanced (>70% in one type)")
    if status_counts.get("verified", 0) == 0:
        warnings.append("dataset contains no human-verified questions; do not use for resume metrics")

    return ValidationReport(
        question_count=len(questions),
        verified_count=status_counts.get("verified", 0),
        candidate_count=status_counts.get("generated_candidate", 0),
        type_counts=dict(sorted(type_counts.items())),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
