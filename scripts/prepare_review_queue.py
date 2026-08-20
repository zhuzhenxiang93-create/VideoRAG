from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from difflib import SequenceMatcher

from video_rag.evaluation.dataset_validation import read_jsonl, validate_questions


OCR_CUES = ("文字", "字样", "字幕", "标题", "标志", "数字", "写着", "显示的名称", "屏幕上")
GENERIC = ("视频中", "这段视频", "画面中", "是否", "什么", "的是", "提到", "显示", "内容")


def normalized(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", value).lower()


def meaningful_chunks(question: str) -> set[str]:
    text = normalized(question)
    for value in GENERIC:
        text = text.replace(normalized(value), "")
    return {text[index : index + 4] for index in range(max(0, len(text) - 3))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a risk-flagged human-review queue without verifying anything.")
    parser.add_argument("--candidates", type=Path, default=Path("data/evaluation/questions.zh.candidates.v1.jsonl"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.supplement_a.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/questions.zh.review_queue.v1.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation/review_queue_v1_report.json"))
    args = parser.parse_args()
    questions = read_jsonl(args.candidates)
    segments = read_jsonl(args.segments)
    segment_by_id = {item["segment_id"]: item for item in segments}
    video_text: dict[str, str] = {}
    for segment in segments:
        video_text[segment["video_id"]] = video_text.get(segment["video_id"], "") + " " + segment.get("transcript", "") + " " + segment.get("visual_caption", "")
    prepared = []
    seen_questions: list[tuple[str, str]] = []
    for item in questions:
        item.pop("_line_number", None)
        flags = []
        question = item["question"].strip()
        if not question.endswith(("？", "?")):
            flags.append("not_interrogative_sentence")
        if len(normalized(question)) < 6:
            flags.append("question_too_short")
        relevant_text = " ".join(
            segment_by_id[value].get("transcript", "") + " " + segment_by_id[value].get("visual_caption", "")
            for value in item["relevant_segment_ids"] if value in segment_by_id
        )
        if item["answerable"] and normalized(item["answer"]) not in normalized(relevant_text):
            flags.append("canonical_answer_not_literal_in_source_fields")
        if item["question_type"] == "visual" and any(cue in question for cue in OCR_CUES):
            item["question_type"] = "ocr"
            item["annotation_source"] += "; heuristic_ocr_reclassification_pending_frame_review"
            flags.append("ocr_answer_must_be_checked_on_original_frame")
        if item["question_type"] == "multimodal":
            source = segment_by_id.get(item.get("generation_metadata", {}).get("source_segment_id", ""), {})
            transcript, caption = normalized(source.get("transcript", "")), normalized(source.get("visual_caption", ""))
            answer = normalized(item["answer"])
            if not transcript or not caption or (answer and (answer in transcript or answer in caption)):
                flags.append("multimodal_joint_evidence_not_demonstrated")
        if not item["answerable"]:
            chunks = meaningful_chunks(question)
            overlap = sorted(chunk for chunk in chunks if chunk and chunk in normalized(video_text.get(item["video_id"], "")))
            if overlap:
                flags.append("unknown_may_be_answerable_in_video")
                item["review_metadata"] = {"matched_question_chunks": overlap[:10]}
        normalized_question = normalized(question)
        near = [
            question_id for question_id, previous in seen_questions
            if SequenceMatcher(None, normalized_question, previous).ratio() >= 0.88
        ]
        if near:
            flags.append("near_duplicate_question")
            item.setdefault("review_metadata", {})["near_duplicate_ids"] = near[:5]
        seen_questions.append((item["question_id"], normalized_question))
        item["review_flags"] = flags
        item["review_priority"] = "high_quality_first" if not flags else "needs_attention"
        prepared.append(item)

    prepared.sort(key=lambda item: (len(item["review_flags"]), item["question_type"], item["question_id"]))
    report = validate_questions(prepared, segments)
    if not report.valid:
        raise ValueError("review queue failed validation: " + "; ".join(report.errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in prepared), encoding="utf-8")
    flag_counts = Counter(flag for item in prepared for flag in item["review_flags"])
    summary = {
        **report.to_dict(), "status": "generated_candidate_review_queue", "formal_metric_eligible": False,
        "source_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "no_flag_count": sum(not item["review_flags"] for item in prepared),
        "needs_attention_count": sum(bool(item["review_flags"]) for item in prepared),
        "flag_counts": dict(flag_counts),
        "note": "flags are triage heuristics, not verification decisions; every accepted item still requires video playback",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
