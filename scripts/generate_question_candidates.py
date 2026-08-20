from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from video_rag.adapters import QwenVLService
from video_rag.evaluation.dataset_validation import read_jsonl, validate_questions


TARGETS = {"audio": 50, "visual": 45, "multimodal": 40, "unknown_route": 40}


def parse_json_array(text: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("model output contains no JSON array")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("model output is not a JSON array")
    return value


def round_robin_segments(segments: list[dict], predicate) -> list[dict]:
    by_video: defaultdict[str, list[dict]] = defaultdict(list)
    for item in segments:
        if predicate(item):
            by_video[item["video_id"]].append(item)
    for values in by_video.values():
        values.sort(key=lambda item: item["start_time"])
    result = []
    while any(by_video.values()):
        for video_id in sorted(by_video):
            if by_video[video_id]:
                result.append(by_video[video_id].pop(0))
    return result


def prompt_for(question_type: str, rows: list[dict]) -> str:
    sources = []
    for index, item in enumerate(rows):
        if question_type == "audio":
            evidence = item.get("transcript", "")
        elif question_type == "visual":
            evidence = item.get("visual_caption", "")
        else:
            evidence = f"ASR: {item.get('transcript','')}\n视觉描述: {item.get('visual_caption','')}"
        sources.append({"source_id": index, "evidence": evidence[:1800]})
    rule = {
        "audio": "问题必须只靠ASR可回答，答案必须是ASR直接支持的短语。",
        "visual": "问题必须观察原画面才能回答，视觉描述只用于生成候选，之后必须人工核对原帧。",
        "multimodal": "问题必须联合ASR和画面信息才能唯一回答，任一单路单独都不够。",
        "unknown_route": "为每个目标视频写一个视频证据中不应有答案的问题；不要根据常识给答案。",
    }[question_type]
    unknown_rule = '无答案题的answer必须是空字符串，answer_aliases必须是空数组。' if question_type == "unknown_route" else "answer_aliases必须包含answer本身。"
    return (
        "你在为中文视频问答项目生成待人工复核候选，不能把候选称为人工标注。"
        f"{rule}{unknown_rule} 每个source恰好生成一个问题。不要输出解释，只输出JSON数组，"
        "每项字段只能是source_id、question、answer、answer_aliases。问题自然、具体、答案唯一；"
        "不要出现segment、source_id、文件名等内部信息。\n来源：\n"
        + json.dumps(sources, ensure_ascii=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a balanced candidate pool; never marks items verified.")
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.supplement_a.jsonl"))
    parser.add_argument("--seed", type=Path, default=Path("data/evaluation/questions.zh.seed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/questions.zh.candidates.v1.jsonl"))
    parser.add_argument("--raw-log", type=Path, default=Path("artifacts/evaluation/question_candidate_generation_raw.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation/question_candidates_v1_report.json"))
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()
    segments = read_jsonl(args.segments)
    for item in segments:
        item.pop("_line_number", None)
    seed = read_jsonl(args.seed)
    for item in seed:
        item.pop("_line_number", None)
        item["verification_status"] = "generated_candidate"
    counts = Counter(item["question_type"] for item in seed)
    pools = {
        "audio": round_robin_segments(segments, lambda item: bool(item.get("transcript", "").strip())),
        "visual": round_robin_segments(segments, lambda item: bool(item.get("keyframes")) and bool(item.get("visual_caption", "").strip())),
        "multimodal": round_robin_segments(segments, lambda item: bool(item.get("transcript", "").strip()) and bool(item.get("visual_caption", "").strip())),
        "unknown_route": round_robin_segments(segments, lambda item: True),
    }
    requests = []
    for question_type, target in TARGETS.items():
        needed = max(0, target - counts[question_type])
        pool = pools[question_type]
        if needed > len(pool):
            raise ValueError(f"not enough source segments for {question_type}: need {needed}, have {len(pool)}")
        for offset in range(0, needed, args.batch_size):
            requests.append((question_type, pool[offset : min(needed, offset + args.batch_size)]))

    completed: dict[str, dict] = {}
    if args.raw_log.exists():
        for item in read_jsonl(args.raw_log):
            completed[item["batch_id"]] = item
    service = QwenVLService(args.model, max_new_tokens=1600)
    generated = []
    args.raw_log.parent.mkdir(parents=True, exist_ok=True)
    for batch_number, (question_type, rows) in enumerate(requests):
        source_ids = [item["segment_id"] for item in rows]
        legacy_generation_type = "unknown" if question_type == "unknown_route" else question_type
        batch_id = hashlib.sha256((legacy_generation_type + "\0" + "\0".join(source_ids)).encode()).hexdigest()[:20]
        if batch_id in completed:
            parsed = completed[batch_id]["parsed"]
        else:
            prompt = prompt_for(question_type, rows)
            raw = service.infer([{"type": "text", "text": prompt}], max_new_tokens=1600)
            parsed = parse_json_array(raw)
            event = {"batch_id": batch_id, "batch_number": batch_number, "question_type": question_type, "source_segment_ids": source_ids, "raw": raw, "parsed": parsed}
            with args.raw_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
        by_source = {int(item["source_id"]): item for item in parsed}
        for source_id, segment in enumerate(rows):
            value = by_source.get(source_id)
            if value is None:
                continue
            answerable = question_type != "unknown_route"
            answer = str(value.get("answer", "")).strip() if answerable else ""
            aliases = [str(item).strip() for item in value.get("answer_aliases", []) if str(item).strip()] if answerable else []
            if answer and answer not in aliases:
                aliases.insert(0, answer)
            question_id = f"cand_{question_type}_{len(generated)+1:04d}"
            generated.append({
                "question_id": question_id, "question": str(value.get("question", "")).strip(),
                "answer": answer, "answer_aliases": list(dict.fromkeys(aliases)), "question_type": question_type,
                "video_id": segment["video_id"],
                "relevant_segment_ids": [segment["segment_id"]] if answerable else [],
                "evidence_start": float(segment["start_time"]) if answerable else None,
                "evidence_end": float(segment["end_time"]) if answerable else None,
                "answerable": answerable, "verification_status": "generated_candidate",
                "annotation_source": f"model_generated_from_{question_type}_evidence_pending_human_review",
                "generation_metadata": {"model": args.model, "batch_id": batch_id, "source_segment_id": segment["segment_id"]},
            })
    service.unload()
    existing_questions = {item["question"].strip() for item in seed}
    unique_generated = []
    for item in generated:
        if item["question"] and item["question"] not in existing_questions and item["answerable"] == bool(item["answer"]):
            existing_questions.add(item["question"])
            unique_generated.append(item)
    all_questions = seed + unique_generated
    report = validate_questions(all_questions, segments)
    if not report.valid:
        raise ValueError("generated pool failed schema validation: " + "; ".join(report.errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in all_questions), encoding="utf-8")
    summary = {
        **report.to_dict(), "output": str(args.output), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "target_counts": TARGETS, "formal_metric_eligible": False,
        "ocr_gap": "0 OCR candidates by design because no independent OCR evidence exists yet",
        "human_review_required": True, "raw_generation_log": str(args.raw_log),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
