from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_rag.config import load_config
from video_rag.evaluation import evaluate_retrieval
from video_rag.retrieval import (
    ClipVisionRetriever,
    InMemoryLexicalRetriever,
    QwenTextRetriever,
    reciprocal_rank_fusion,
)
from video_rag.storage import load_segments


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recall routes and RRF ablations.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--index-dir", default=Path("artifacts/indexes"), type=Path)
    parser.add_argument("--config", default=Path("config.toml"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/retrieval_metrics.json"), type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    segments = load_segments(args.segments)
    questions = load_questions(args.questions)
    retrievers = {
        "bm25": InMemoryLexicalRetriever(),
        "embedding": QwenTextRetriever(
            config.models.text_embedding, device=args.device, index_dir=args.index_dir
        ),
        "clip": ClipVisionRetriever(
            config.models.clip, device=args.device, index_dir=args.index_dir
        ),
    }
    for retriever in retrievers.values():
        retriever.build(segments)

    ground_truth = {
        item["question_id"]: item["relevant_segment_ids"]
        for item in questions
    }
    route_predictions: dict[str, dict[str, list[str]]] = {
        name: {} for name in retrievers
    }
    route_predictions["rrf"] = {}
    for item in questions:
        question_id = item["question_id"]
        route_hits = []
        for name, retriever in retrievers.items():
            hits = retriever.search(item["question"], 20)
            route_hits.append(hits)
            route_predictions[name][question_id] = [hit.segment_id for hit in hits]
        route_predictions["rrf"][question_id] = [
            hit.segment_id for hit in reciprocal_rank_fusion(route_hits, top_k=20)
        ]

    results = {
        name: evaluate_retrieval(predictions, ground_truth)
        for name, predictions in route_predictions.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

