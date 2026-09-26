"""Evaluate cross-video retrieval, model reranking, and optional full answers.

Question labels are consulted only after inference. No video_id is passed to search/ask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from run_server import build_real_pipeline

from video_rag.evaluation.answer import token_f1
from video_rag.evaluation.tutorialvqa import latency_summary, overlaps_reference
from video_rag.storage import load_segments


def video_hit(evidence, video_id: str, k: int) -> bool:
    return any(item.segment.video_id == video_id for item in evidence[:k])


def segment_hit(evidence, video_id: str, intervals: list[dict], k: int) -> bool:
    return any(
        item.segment.video_id == video_id and overlaps_reference(item.segment, intervals)
        for item in evidence[:k]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("artifacts/tutorialvqa/scale-76/evaluation/test.jsonl"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/tutorialvqa/scale-76/segments.ocr.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/tutorialvqa/scale-76/indexes-en-clip"))
    parser.add_argument("--config", type=Path, default=Path("config.tutorialvqa.full-demo.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tutorialvqa/scale-76/metrics-global"))
    parser.add_argument("--limit", type=int, default=0, help="First N questions; 0 means all")
    parser.add_argument("--generate-sample", type=int, default=0, help="Evenly spaced question sample")
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.open() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise ValueError("No questions selected")
    if args.generate_sample < 0:
        raise ValueError("generate-sample must be non-negative")
    generated_positions = set()
    if args.generate_sample:
        count = min(args.generate_sample, len(questions))
        generated_positions = {round(i * (len(questions) - 1) / max(count - 1, 1)) for i in range(count)}
    pipeline = build_real_pipeline(
        segments_path=args.segments, index_dir=args.index_dir,
        config_path=args.config, device="cuda",
    )
    pipeline.warmup("software tutorial steps")
    # Exclude one-time model loading from steady-state latency measurements.
    pipeline.search(questions[0]["question"], rerank=True)
    if generated_positions:
        pipeline.ask(questions[0]["question"])
    segments = load_segments(args.segments)
    known_ids = {s.segment_id for s in segments}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "predictions.jsonl"
    totals = {name: [] for name in (
        "baseline_video@1", "baseline_video@5", "reranked_video@1", "reranked_video@5",
        "baseline_segment@1", "baseline_segment@5", "baseline_segment@20",
        "reranked_segment@1", "reranked_segment@5",
        "baseline_ms", "reranked_ms", "answer_ms", "answer_f1", "citation_valid", "answer_evidence_hit",
    )}
    answered = 0
    with out.open("w") as stream:
        for position, item in enumerate(questions):
            question = item["question"]
            video_id = str(item["video_id"])
            intervals = item["reference_intervals"]
            t0 = perf_counter()
            baseline = pipeline.search(question)
            baseline_ms = (perf_counter() - t0) * 1000
            t0 = perf_counter()
            reranked = pipeline.search(question, rerank=True)
            reranked_ms = (perf_counter() - t0) * 1000
            row = {"question_id": item["question_id"], "reference_video_id": video_id,
                   "baseline_ids": [e.segment.segment_id for e in baseline],
                   "reranked_ids": [e.segment.segment_id for e in reranked],
                   "baseline_ms": baseline_ms, "reranked_ms": reranked_ms}
            for prefix, evidence in (("baseline", baseline), ("reranked", reranked)):
                for k in (1, 5):
                    key = f"{prefix}_video@{k}"
                    row[key] = video_hit(evidence, video_id, k)
                    totals[key].append(row[key])
                    key = f"{prefix}_segment@{k}"
                    row[key] = segment_hit(evidence, video_id, intervals, k)
                    totals[key].append(row[key])
            row["baseline_segment@20"] = segment_hit(baseline, video_id, intervals, 20)
            totals["baseline_segment@20"].append(row["baseline_segment@20"])
            totals["baseline_ms"].append(baseline_ms)
            totals["reranked_ms"].append(reranked_ms)
            if position in generated_positions:
                answer = pipeline.ask(question)
                citations = [e.segment for e in answer.evidence]
                row.update({"answer": answer.answer, "abstained": answer.abstained,
                            "citations": list(answer.citations),
                            "answer_ms": answer.latency_ms["total"],
                            "answer_f1": token_f1(answer.answer, item["reference_answer"]),
                            "citation_valid": bool(answer.citations) and set(answer.citations).issubset(known_ids)
                            and set(answer.citations) == {s.segment_id for s in citations},
                            "answer_evidence_hit": any(s.video_id == video_id and overlaps_reference(s, intervals) for s in citations)})
                answered += not answer.abstained
                for key in ("answer_ms", "answer_f1", "citation_valid", "answer_evidence_hit"):
                    totals[key].append(row[key])
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if (position + 1) % 25 == 0 or position + 1 == len(questions):
                print(f"[{position + 1}/{len(questions)}] global evaluation", flush=True)
    report = {"dataset": "TutorialVQA", "split": questions[0].get("split"),
              "question_count": len(questions), "video_count": len({q["video_id"] for q in questions}),
              "scope": "all indexed videos; no ground-truth video_id passed to inference",
              "relevance": "any positive overlap with official answer interval",
              "reranker": "Qwen3-Reranker-0.6B text model over ASR, OCR and captions",
              "retrieval": {key: mean(values) for key, values in totals.items() if values and "@" in key},
              "latency_ms": {key: latency_summary(totals[key]) for key in ("baseline_ms", "reranked_ms", "answer_ms") if totals[key]},
              "generation": {"count": len(totals["answer_f1"]), "answered": answered,
                             "token_f1": mean(totals["answer_f1"]) if totals["answer_f1"] else None,
                             "citation_valid_rate": mean(totals["citation_valid"]) if totals["citation_valid"] else None,
                             "evidence_hit_rate": mean(totals["answer_evidence_hit"]) if totals["answer_evidence_hit"] else None}}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
