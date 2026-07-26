from __future__ import annotations

import argparse
from pathlib import Path

from video_rag.adapters import Qwen3Reranker, QwenVLEvidenceGenerator, QwenVLService
from video_rag.api import create_app
from video_rag.config import load_config
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval import ClipVisionRetriever, InMemoryLexicalRetriever, QwenTextRetriever
from video_rag.storage import load_segments


def build_real_pipeline(
    *,
    segments_path: Path,
    index_dir: Path,
    config_path: Path,
    device: str = "cuda",
    low_vram: bool = False,
) -> VideoRAGPipeline:
    config = load_config(config_path)
    service = QwenVLService(config.models.vision_language)
    pipeline = VideoRAGPipeline(
        retrievers=[
            InMemoryLexicalRetriever(),
            QwenTextRetriever(
                config.models.text_embedding,
                device=device,
                index_dir=index_dir,
            ),
            ClipVisionRetriever(
                config.models.clip,
                device=device,
                index_dir=index_dir,
            ),
        ],
        reranker=Qwen3Reranker(
            config.models.reranker,
            unload_after_score=low_vram,
        ),
        generator=QwenVLEvidenceGenerator(service),
        recall_top_k=max(
            config.retrieval.sparse_top_k,
            config.retrieval.text_top_k,
            config.retrieval.vision_top_k,
        ),
        fusion_top_k=config.retrieval.fusion_top_k,
        rerank_top_k=config.retrieval.rerank_top_k,
        rrf_k=config.retrieval.rrf_k,
        minimum_rerank_score=config.generation.minimum_rerank_score,
    )
    pipeline.build(load_segments(segments_path))
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real multimodal Video RAG API.")
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--index-dir", default=Path("artifacts/indexes"), type=Path)
    parser.add_argument("--config", default=Path("config.toml"), type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Unload the reranker before Qwen-VL generation; slower but suitable for 24GB.",
    )
    args = parser.parse_args()
    pipeline = build_real_pipeline(
        segments_path=args.segments,
        index_dir=args.index_dir,
        config_path=args.config,
        device=args.device,
        low_vram=args.low_vram,
    )
    create_app(pipeline).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
