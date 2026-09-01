from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from video_rag.adapters import Qwen3Reranker, Qwen3VLReranker
from video_rag.config import load_config
from video_rag.evaluation import evaluate_retrieval
from video_rag.retrieval import (
    AdaptiveFusionPolicy,
    BM25Retriever,
    ClipVisionRetriever,
    Qwen3VLEmbeddingRetriever,
    QwenTextRetriever,
    reciprocal_rank_fusion,
)
from video_rag.storage import load_segments

REQUIRED_FIELDS = {
    "question_id",
    "question",
    "answer",
    "relevant_segment_ids",
    "question_type",
}
ROUTES = {
    "bm25": ("bm25",),
    "embedding": ("embedding",),
    "clip": ("clip",),
    "bm25+embedding": ("bm25", "embedding"),
    "bm25+clip": ("bm25", "clip"),
    "embedding+clip": ("embedding", "clip"),
    "all_rrf": ("bm25", "embedding", "clip"),
    "adaptive_rrf": ("bm25", "embedding", "clip"),
}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def load_questions(path: Path, known_segment_ids: set[str]) -> list[dict]:
    questions: list[dict] = []
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = REQUIRED_FIELDS - item.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")
            question_id = str(item["question_id"]).strip()
            if not question_id or question_id in identifiers:
                raise ValueError(f"line {line_number}: duplicate or empty question_id")
            relevant = item["relevant_segment_ids"]
            if not isinstance(relevant, list) or not relevant:
                raise ValueError(f"line {line_number}: relevant_segment_ids must be non-empty")
            unknown = set(relevant) - known_segment_ids
            if unknown:
                raise ValueError(f"line {line_number}: unknown segments {sorted(unknown)}")
            identifiers.add(question_id)
            questions.append(item)
    if not questions:
        raise ValueError("question file must not be empty")
    return questions


def metrics_with_breakdown(
    predictions: dict[str, list[str]], questions: list[dict]
) -> dict[str, object]:
    ground_truth = {item["question_id"]: item["relevant_segment_ids"] for item in questions}
    overall = evaluate_retrieval(predictions, ground_truth)
    by_type: dict[str, dict[str, float]] = {}
    types = sorted({item["question_type"] for item in questions})
    for question_type in types:
        subset = [item for item in questions if item["question_type"] == question_type]
        subset_truth = {item["question_id"]: item["relevant_segment_ids"] for item in subset}
        by_type[question_type] = evaluate_retrieval(predictions, subset_truth)
    return {"overall": overall, "by_question_type": by_type}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible retrieval ablations.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--index-dir", default=Path("artifacts/indexes"), type=Path)
    parser.add_argument("--config", default=Path("config.toml"), type=Path)
    parser.add_argument(
        "--output", default=Path("artifacts/evaluation/retrieval_ablation.json"), type=Path
    )
    parser.add_argument(
        "--csv-output", default=Path("artifacts/evaluation/retrieval_ablation.csv"), type=Path
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--with-reranker", action="store_true")
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    config = load_config(args.config)
    segments = load_segments(args.segments)
    segment_by_id = {item.segment_id: item for item in segments}
    questions = load_questions(args.questions, set(segment_by_id))
    if config.retrieval.vision_backend == "qwen3_vl":
        vision_retriever = Qwen3VLEmbeddingRetriever(
            config.models.qwen3_vl_embedding,
            implementation_repository=config.models.qwen3_vl_repository,
            index_dir=args.index_dir,
            max_frames=config.generation.max_frames,
            fps=config.generation.video_fps,
        )
    else:
        vision_retriever = ClipVisionRetriever(
            config.models.clip, device=args.device, index_dir=args.index_dir
        )
    retrievers = {
        "bm25": BM25Retriever(
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
        ),
        "embedding": QwenTextRetriever(
            config.models.text_embedding, device=args.device, index_dir=args.index_dir
        ),
        "clip": vision_retriever,
    }
    for retriever in retrievers.values():
        retriever.build(segments)
        retriever.search("视频内容", 1)
    fusion_policy = AdaptiveFusionPolicy(
        sparse_source=retrievers["bm25"].name,
        text_source=retrievers["embedding"].name,
        vision_source=retrievers["clip"].name,
        sparse_weight=config.retrieval.sparse_weight,
        text_weight=config.retrieval.text_weight,
        vision_weight=config.retrieval.vision_weight,
        visual_sparse_weight=config.retrieval.visual_sparse_weight,
        visual_text_weight=config.retrieval.visual_text_weight,
        visual_vision_weight=config.retrieval.visual_vision_weight,
        agreement_bonus=config.retrieval.agreement_bonus,
    )

    predictions = {name: {} for name in ROUTES}
    latency = {name: [] for name in ROUTES}
    all_candidates: dict[str, list[str]] = {}
    for item in questions:
        question_id = item["question_id"]
        route_hits = {}
        route_latency = {}
        for name, retriever in retrievers.items():
            started = perf_counter()
            route_hits[name] = retriever.search(item["question"], args.top_k)
            route_latency[name] = (perf_counter() - started) * 1000
        for route_name, component_names in ROUTES.items():
            if len(component_names) == 1:
                ranked = route_hits[component_names[0]]
            else:
                ranked = reciprocal_rank_fusion(
                    [route_hits[name] for name in component_names],
                    k=config.retrieval.rrf_k,
                    top_k=args.top_k,
                    source_weights=(
                        fusion_policy.source_weights(item["question"])
                        if route_name == "adaptive_rrf"
                        else None
                    ),
                    agreement_bonus=(
                        fusion_policy.agreement_bonus
                        if route_name == "adaptive_rrf"
                        else 0.0
                    ),
                )
            predictions[route_name][question_id] = [hit.segment_id for hit in ranked]
            latency[route_name].append(sum(route_latency[name] for name in component_names))
        all_candidates[question_id] = predictions["adaptive_rrf"][question_id]

    if args.with_reranker:
        if config.retrieval.reranker_backend == "qwen3_vl":
            reranker = Qwen3VLReranker(
                config.models.qwen3_vl_reranker,
                implementation_repository=config.models.qwen3_vl_repository,
                max_frames=config.generation.max_frames,
                fps=config.generation.video_fps,
                unload_after_score=args.low_vram,
            )
        else:
            reranker = Qwen3Reranker(config.models.reranker, unload_after_score=args.low_vram)
        predictions["adaptive_rrf+reranker"] = {}
        latency["adaptive_rrf+reranker"] = []
        first_question = questions[0]
        first_candidate = segment_by_id[all_candidates[first_question["question_id"]][0]]
        reranker.score("视频内容", [first_candidate])
        for item in questions:
            question_id = item["question_id"]
            candidates = [segment_by_id[value] for value in all_candidates[question_id]]
            started = perf_counter()
            scores = reranker.score(item["question"], candidates)
            count = len(candidates)
            blended_scores = [
                config.retrieval.reranker_weight * score
                + (1.0 - config.retrieval.reranker_weight)
                * (1.0 - rank / max(1, count - 1))
                for rank, score in enumerate(scores)
            ]
            reranked = sorted(
                zip(candidates, scores, blended_scores, strict=True),
                key=lambda item: item[2],
                reverse=True,
            )
            predictions["adaptive_rrf+reranker"][question_id] = [
                segment.segment_id for segment, _, _ in reranked
            ]
            latency["adaptive_rrf+reranker"].append(
                latency["adaptive_rrf"][len(latency["adaptive_rrf+reranker"])]
                + (perf_counter() - started) * 1000
            )

    results = {}
    for name, route_predictions in predictions.items():
        route_metrics = metrics_with_breakdown(route_predictions, questions)
        timings = latency[name]
        route_metrics["latency_ms"] = {
            "mean": mean(timings),
            "median": median(timings),
            "p95": percentile(timings, 0.95),
        }
        results[name] = route_metrics

    report = {
        "dataset": str(args.questions),
        "question_count": len(questions),
        "top_k": args.top_k,
        "routes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("route", "recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "mean_ms", "p95_ms")
        )
        for name, result in results.items():
            metrics = result["overall"]
            timings = result["latency_ms"]
            writer.writerow(
                (
                    name,
                    metrics["recall@1"],
                    metrics["recall@5"],
                    metrics["recall@10"],
                    metrics["mrr"],
                    metrics["ndcg@5"],
                    timings["mean"],
                    timings["p95"],
                )
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
