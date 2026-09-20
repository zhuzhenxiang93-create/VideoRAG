"""Evaluate scoped TutorialVQA retrieval and grounded answer generation.

Reference answers and intervals are used only after retrieval/generation for scoring.
They are never added to the model prompt or search index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from run_server import build_real_pipeline
from video_rag.evaluation import evaluate_answers, evaluate_retrieval
from video_rag.evaluation.answer import exact_match, token_f1
from video_rag.evaluation.tutorialvqa import (
    best_temporal_iou,
    latency_summary,
    overlaps_reference,
    relevant_segment_ids,
)
from video_rag.storage import load_segments


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TutorialVQA with temporal labels.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tutorialvqa/metrics"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/tutorialvqa/bounded-v2/segments.ocr.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/tutorialvqa/bounded-v2/indexes"))
    parser.add_argument("--config", type=Path, default=Path("config.tutorialvqa.toml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()
    if args.top_k < 10:
        raise ValueError("top-k must be at least 10 for Recall@10")

    questions = read_jsonl(args.questions)
    segments = load_segments(args.segments)
    segment_by_id = {segment.segment_id: segment for segment in segments}
    known_videos = {segment.video_id for segment in segments}
    for item in questions:
        if item["video_id"] not in known_videos:
            raise ValueError(f"Unknown video_id in evaluation: {item['video_id']}")
        if not item.get("reference_intervals") or not item.get("reference_answer"):
            raise ValueError(f"Missing reference labels: {item['question_id']}")

    pipeline = build_real_pipeline(
        segments_path=args.segments,
        index_dir=args.index_dir,
        config_path=args.config,
        device=args.device,
        low_vram=args.low_vram,
    )
    pipeline.warmup("tutorial steps and visible interface text")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = questions[0].get("split", args.questions.stem) if questions else args.questions.stem
    predictions_path = args.output_dir / f"{split}-predictions.jsonl"
    report_path = args.output_dir / f"{split}-report.json"

    retrieval_predictions: dict[str, list[str]] = {}
    retrieval_truth: dict[str, list[str]] = {}
    retrieval_latency: list[float] = []
    temporal_iou = {1: [], 5: [], 10: []}
    scope_violations = 0
    rows: list[dict] = []

    for position, item in enumerate(questions, start=1):
        question_id = item["question_id"]
        video_id = item["video_id"]
        intervals = item["reference_intervals"]
        started = perf_counter()
        retrieved = pipeline.search(item["question"], [video_id])[: args.top_k]
        retrieval_ms = (perf_counter() - started) * 1000
        retrieved_segments = [evidence.segment for evidence in retrieved]
        retrieved_ids = [segment.segment_id for segment in retrieved_segments]
        relevant_ids = relevant_segment_ids(segments, video_id, intervals)
        if not relevant_ids:
            raise ValueError(f"No indexed segment overlaps references for {question_id}")
        retrieval_predictions[question_id] = retrieved_ids
        retrieval_truth[question_id] = relevant_ids
        retrieval_latency.append(retrieval_ms)
        scope_violations += sum(segment.video_id != video_id for segment in retrieved_segments)
        for cutoff in temporal_iou:
            temporal_iou[cutoff].append(best_temporal_iou(retrieved_segments[:cutoff], intervals))

        row = {
            "question_id": question_id,
            "split": split,
            "video_id": video_id,
            "question": item["question"],
            "retrieved_segment_ids": retrieved_ids,
            "relevant_segment_ids": relevant_ids,
            "retrieval_ms": retrieval_ms,
        }
        if args.generate:
            answer = pipeline.ask(item["question"], [video_id])
            cited_segments = [segment_by_id[value] for value in answer.citations if value in segment_by_id]
            cited_overlap = [overlaps_reference(segment, intervals) for segment in cited_segments]
            evidence_ids = [evidence.segment.segment_id for evidence in answer.evidence]
            citation_valid = bool(answer.citations) and set(answer.citations) == set(evidence_ids)
            citation_scoped = all(segment.video_id == video_id for segment in cited_segments)
            row.update(
                {
                    "answer": answer.answer,
                    "abstained": answer.abstained,
                    "confidence": answer.confidence,
                    "citations": list(answer.citations),
                    "cited_intervals": [
                        {
                            "segment_id": segment.segment_id,
                            "start_time": segment.start_time,
                            "end_time": segment.end_time,
                        }
                        for segment in cited_segments
                    ],
                    "citation_valid": citation_valid,
                    "citation_scoped": citation_scoped,
                    "evidence_hit": any(cited_overlap),
                    "evidence_precision": mean(cited_overlap) if cited_overlap else 0.0,
                    "exact_match": exact_match(answer.answer, item["reference_answer"]),
                    "token_f1": token_f1(answer.answer, item["reference_answer"]),
                    "latency_ms": answer.latency_ms,
                }
            )
        rows.append(row)
        write_jsonl(predictions_path, rows)
        print(
            f"[{position}/{len(questions)}] {question_id} "
            f"retrieval_hit={bool(set(retrieved_ids[:1]) & set(relevant_ids))} "
            f"answered={('n/a' if not args.generate else row.get('abstained') is False)}",
            flush=True,
        )

    report = {
        "dataset": "TutorialVQA",
        "split": split,
        "question_count": len(questions),
        "video_count": len({item["video_id"] for item in questions}),
        "retrieval": {
            **evaluate_retrieval(retrieval_predictions, retrieval_truth),
            "mean_best_temporal_iou@1": mean(temporal_iou[1]),
            "mean_best_temporal_iou@5": mean(temporal_iou[5]),
            "mean_best_temporal_iou@10": mean(temporal_iou[10]),
            "scope_violation_count": scope_violations,
            "scope_violation_rate": scope_violations / max(1, sum(len(v) for v in retrieval_predictions.values())),
            "latency_ms": latency_summary(retrieval_latency),
        },
        "configuration": {
            "segments": str(args.segments),
            "index_dir": str(args.index_dir),
            "config": str(args.config),
            "top_k": args.top_k,
            "scoped_by_video_id": True,
            "reference_labels_used_after_inference_only": True,
        },
    }
    if args.generate:
        references = {item["question_id"]: item["reference_answer"] for item in questions}
        predicted_answers = {row["question_id"]: row["answer"] for row in rows}
        answered = [row for row in rows if not row["abstained"]]
        answer_latencies = [row["latency_ms"]["total"] for row in rows]
        generation_latencies = [row["latency_ms"]["generation"] for row in rows]
        answer_metrics = evaluate_answers(predicted_answers, references)
        report["answer_generation"] = {
            **answer_metrics,
            "answered_count": len(answered),
            "answered_rate": len(answered) / len(rows),
            "citation_valid_rate_answered": mean(row["citation_valid"] for row in answered) if answered else 0.0,
            "citation_scoped_rate_answered": mean(row["citation_scoped"] for row in answered) if answered else 0.0,
            "evidence_hit_rate_answered": mean(row["evidence_hit"] for row in answered) if answered else 0.0,
            "mean_evidence_precision_answered": mean(row["evidence_precision"] for row in answered) if answered else 0.0,
            "grounded_answer_rate": mean(
                (not row["abstained"])
                and row["citation_valid"]
                and row["citation_scoped"]
                and row["evidence_hit"]
                for row in rows
            ),
            "answer_latency_ms": latency_summary(answer_latencies),
            "generation_latency_ms": latency_summary(generation_latencies),
        }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
