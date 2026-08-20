from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from video_rag.config import load_config
from video_rag.evaluation import evaluate_retrieval
from video_rag.retrieval import ClipVisionRetriever, FrameClipVisionRetriever
from video_rag.schemas import SearchHit
from video_rag.storage import load_segments


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def metrics(predictions: dict[str, list[str]], questions: list[dict]) -> dict:
    truth = {item["question_id"]: item["relevant_segment_ids"] for item in questions}
    overall = evaluate_retrieval(predictions, truth)
    visual = [item for item in questions if item.get("question_type") == "visual"]
    visual_truth = {item["question_id"]: item["relevant_segment_ids"] for item in visual}
    return {
        "overall_generated_candidate": overall,
        "visual_generated_candidate": evaluate_retrieval(predictions, visual_truth) if visual else None,
        "visual_question_count": len(visual),
    }


class ExistingSegmentIndex:
    def __init__(self, index_dir: Path, encoder: ClipVisionRetriever) -> None:
        import faiss

        self.encoder = encoder
        self.index = faiss.read_index(str(index_dir / "vision_segment_mean_unique_zh.faiss"))
        self.segment_ids = json.loads(
            (index_dir / "vision_segment_mean_unique_zh.json").read_text(encoding="utf-8")
        )["segment_ids"]

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        scores, indices = self.index.search(self.encoder.encode_query(query), min(top_k, len(self.segment_ids)))
        return [
            SearchHit(self.segment_ids[int(index)], float(score), "vision_segment_mean_unique_zh", rank)
            for rank, (index, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled E0/E1/E2 visual retrieval diagnostic.")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.zh.seed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/visual_e0_e1_e2.json"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--depths", type=int, nargs="+", default=[20, 50, 100, 145])
    args = parser.parse_args()

    config = load_config(args.config)
    segments = load_segments(args.segments)
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    e0 = ClipVisionRetriever(config.models.clip, device="cuda", index_dir=args.index_dir)
    e0.build(segments)
    e0.search("预热视觉检索", 1)

    experiments = [("E0_segment_mean", e0), ("E0_unique_frame_cache", ExistingSegmentIndex(args.index_dir, e0))]
    frame_retrievers = []
    for aggregation in ("max", "top2_mean"):
        for depth in args.depths:
            retriever = FrameClipVisionRetriever(
                config.models.clip,
                device="cuda",
                index_dir=args.index_dir,
                aggregation=aggregation,
                frame_candidate_k=depth,
            )
            retriever.load(segments_path=args.segments)
            retriever._model = e0._model
            retriever._processor = e0._processor
            frame_retrievers.append(retriever)
            label = f"{'E1_frame_max' if aggregation == 'max' else 'E2_frame_top2'}_depth{depth}"
            experiments.append((label, retriever))
    for _, retriever in experiments[1:]:
        retriever.search("预热视觉检索", 1)

    results = {}
    for label, retriever in experiments:
        predictions = {}
        timings = []
        returned_counts = []
        for item in questions:
            started = perf_counter()
            hits = retriever.search(item["question"], args.top_k)
            timings.append((perf_counter() - started) * 1000)
            predictions[item["question_id"]] = [hit.segment_id for hit in hits]
            returned_counts.append(len({hit.segment_id for hit in hits}))
        result = metrics(predictions, questions)
        result["latency_ms"] = {
            "samples": len(timings),
            "mean": mean(timings),
            "p50": median(timings),
            "p95": percentile(timings, 0.95),
            "p99": percentile(timings, 0.99),
            "max": max(timings),
        }
        result["returned_unique_segments"] = {
            "min": min(returned_counts), "mean": mean(returned_counts), "max": max(returned_counts)
        }
        result["predictions"] = predictions
        results[label] = result

    old_predictions = results["E0_segment_mean"]["predictions"]
    unique_predictions = results["E0_unique_frame_cache"]["predictions"]
    parity = {
        "question_count": len(questions),
        "top1_equal": sum(old_predictions[key][:1] == unique_predictions[key][:1] for key in old_predictions),
        "top10_order_exact": sum(old_predictions[key] == unique_predictions[key] for key in old_predictions),
        "top10_set_equal": sum(set(old_predictions[key]) == set(unique_predictions[key]) for key in old_predictions),
    }
    report = {
        "status": "diagnostic_only",
        "formal_metric_eligible": False,
        "reason": "all 20 questions are generated_candidate and only 2 are visual; no human-verified claim is allowed",
        "controlled_variables": {
            "questions": str(args.questions),
            "question_count": len(questions),
            "segments_sha256": __import__("hashlib").sha256(args.segments.read_bytes()).hexdigest(),
            "model": config.models.clip,
            "hardware": "same server/GPU in one process",
            "segment_top_k": args.top_k,
            "frame_depths": args.depths,
            "boundary_policy": "start <= timestamp < end; no outside-boundary reuse",
        },
        "aggregation_definitions": {
            "E0": "mean all frame embeddings per segment, then normalize (pre-existing index)",
            "E0_unique": "same E0 formula reconstructed from each physical frame encoded once plus its raw feature norm",
            "E1": "max frame similarity for each segment; a shared frame gives the same score to each membership",
            "E2": "mean of best 2 frame similarities; one-frame segments use the single score",
            "tie_break": "segment_id ascending",
        },
        "e0_ranking_parity": parity,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "reason": report["reason"], "results": {
        key: {name: value for name, value in item.items() if name != "predictions"}
        for key, item in results.items()
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
