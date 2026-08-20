from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from scripts.evaluate_visual_granularity import ExistingSegmentIndex, metrics, percentile
from video_rag.config import load_config
from video_rag.retrieval import ClipVisionRetriever, FrameClipVisionRetriever
from video_rag.storage import load_segments


def best_rank(prediction: list[str], relevant: list[str]) -> int | None:
    positions = [prediction.index(item) + 1 for item in relevant if item in prediction]
    return min(positions) if positions else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled E0-E4 supplement-A visual diagnostic.")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.jsonl"))
    parser.add_argument("--supplement-segments", type=Path, default=Path("artifacts/segments.supplement_a.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--supplement-index-dir", type=Path, default=Path("artifacts/indexes_supplement_a"))
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.zh.seed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/visual_e0_e4.json"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    original = load_segments(args.segments)
    supplemented = load_segments(args.supplement_segments)
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    encoder = ClipVisionRetriever(config.models.clip, device="cuda", index_dir=args.index_dir)
    encoder.build(original)
    encoder.search("预热视觉检索", 1)

    e1 = FrameClipVisionRetriever(config.models.clip, index_dir=args.index_dir, aggregation="max", frame_candidate_k=20)
    e1.load(segments_path=args.segments)
    e2 = FrameClipVisionRetriever(config.models.clip, index_dir=args.index_dir, aggregation="top2_mean", frame_candidate_k=145)
    e2.load(segments_path=args.segments)
    e3 = ExistingSegmentIndex(args.supplement_index_dir, encoder)
    e4max = FrameClipVisionRetriever(config.models.clip, index_dir=args.supplement_index_dir, aggregation="max", frame_candidate_k=20)
    e4max.load(segments_path=args.supplement_segments)
    e4top2 = FrameClipVisionRetriever(config.models.clip, index_dir=args.supplement_index_dir, aggregation="top2_mean", frame_candidate_k=165)
    e4top2.load(segments_path=args.supplement_segments)
    experiments = [
        ("E0_original_segment_mean", encoder),
        ("E1_original_frame_max_depth20", e1),
        ("E2_original_frame_top2_exact", e2),
        ("E3_supplement_a_segment_mean", e3),
        ("E4_supplement_a_frame_max_depth20", e4max),
        ("E4_supplement_a_frame_top2_exact", e4top2),
    ]
    for _, retriever in experiments[1:]:
        if isinstance(retriever, FrameClipVisionRetriever):
            retriever._model, retriever._processor = encoder._model, encoder._processor
        retriever.search("预热视觉检索", 1)

    results = {}
    all_predictions = {}
    for label, retriever in experiments:
        predictions = {}
        timings = []
        for item in questions:
            started = perf_counter()
            hits = retriever.search(item["question"], args.top_k)
            timings.append((perf_counter() - started) * 1000)
            predictions[item["question_id"]] = [hit.segment_id for hit in hits]
        all_predictions[label] = predictions
        result = metrics(predictions, questions)
        result["latency_ms"] = {
            "mean": mean(timings), "p50": median(timings), "p95": percentile(timings, .95),
            "p99": percentile(timings, .99), "max": max(timings), "samples": len(timings),
        }
        results[label] = result

    baseline = all_predictions["E0_original_segment_mean"]
    failure_deltas = {}
    for label, predictions in all_predictions.items():
        if label == "E0_original_segment_mean":
            continue
        improved, regressed, unchanged = [], [], []
        for item in questions:
            question_id = item["question_id"]
            old = best_rank(baseline[question_id], item["relevant_segment_ids"])
            new = best_rank(predictions[question_id], item["relevant_segment_ids"])
            old_value, new_value = old or 999, new or 999
            record = {"question_id": question_id, "question_type": item.get("question_type"), "old_rank": old, "new_rank": new}
            (improved if new_value < old_value else regressed if new_value > old_value else unchanged).append(record)
        failure_deltas[label] = {"improved": improved, "regressed": regressed, "unchanged_count": len(unchanged)}

    report = {
        "status": "diagnostic_only",
        "formal_metric_eligible": False,
        "reason": "20 generated_candidate questions (only 2 visual) cannot support a resume claim",
        "control": {
            "model": config.models.clip, "same_gpu_process": True, "top_k": args.top_k,
            "E0_E1_E2_frames": 145, "E3_E4_frames": 165,
            "supplement_qwen_vl_calls": 0, "strict_boundary_membership": True,
        },
        "results": results,
        "rank_deltas_vs_E0": failure_deltas,
        "predictions": all_predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "results": results, "rank_delta_counts": {
        label: {"improved": len(item["improved"]), "regressed": len(item["regressed"]), "unchanged": item["unchanged_count"]}
        for label, item in failure_deltas.items()
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
