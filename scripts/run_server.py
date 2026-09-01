from __future__ import annotations

import argparse
from pathlib import Path

from video_rag.adapters import (
    FusionOrderReranker,
    Qwen3Reranker,
    Qwen3VLReranker,
    Qwen3VLService,
    QwenVLEvidenceGenerator,
    QwenVLService,
)
from video_rag.api import create_app
from video_rag.config import load_config
from video_rag.index_manifest import validate_manifest, validate_runtime_manifest
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval import (
    AdaptiveFusionPolicy,
    BM25Retriever,
    ClipVisionRetriever,
    InMemoryLexicalRetriever,
    OCRBM25Retriever,
    Qwen3VLEmbeddingRetriever,
    QwenTextRetriever,
)
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
    if config.retrieval.vision_backend == "chinese_clip":
        validate_manifest(
            segments_path=segments_path,
            index_dir=index_dir,
            text_model=config.models.text_embedding,
            clip_model=config.models.clip,
        )
    else:
        validate_runtime_manifest(
            segments_path=segments_path,
            index_dir=index_dir,
            index_models={
                "text_dense": config.models.text_embedding,
                "vision_multimodal_qwen3_vl": config.models.qwen3_vl_embedding,
            },
        )

    sparse_retriever = (
        BM25Retriever(k1=config.retrieval.bm25_k1, b=config.retrieval.bm25_b)
        if config.retrieval.sparse_backend == "bm25"
        else InMemoryLexicalRetriever()
    )
    if config.retrieval.vision_backend == "qwen3_vl":
        if not config.models.qwen3_vl_repository:
            raise ValueError(
                "models.qwen3_vl_repository is required when vision_backend='qwen3_vl'"
            )
        vision_retriever = Qwen3VLEmbeddingRetriever(
            config.models.qwen3_vl_embedding,
            implementation_repository=config.models.qwen3_vl_repository,
            index_dir=index_dir,
            max_frames=config.generation.max_frames,
            fps=config.generation.video_fps,
        )
    else:
        vision_retriever = ClipVisionRetriever(
            config.models.clip,
            device=device,
            index_dir=index_dir,
        )

    if config.retrieval.reranker_backend == "fusion_only":
        reranker = FusionOrderReranker()
    elif config.retrieval.reranker_backend == "qwen3_vl":
        if not config.models.qwen3_vl_repository:
            raise ValueError(
                "models.qwen3_vl_repository is required when reranker_backend='qwen3_vl'"
            )
        reranker = Qwen3VLReranker(
            config.models.qwen3_vl_reranker,
            implementation_repository=config.models.qwen3_vl_repository,
            max_frames=config.generation.max_frames,
            fps=config.generation.video_fps,
            unload_after_score=low_vram,
        )
    else:
        reranker = Qwen3Reranker(
            config.models.reranker,
            unload_after_score=low_vram,
        )

    service = (
        Qwen3VLService(config.models.qwen3_vl_generation)
        if config.generation.backend == "qwen3_vl"
        else QwenVLService(config.models.vision_language)
    )
    text_retriever = QwenTextRetriever(
        config.models.text_embedding,
        device=device,
        index_dir=index_dir,
    )
    ocr_retriever = (
        OCRBM25Retriever(k1=config.retrieval.bm25_k1, b=config.retrieval.bm25_b)
        if config.ocr.enabled
        else None
    )
    retrievers = [sparse_retriever, text_retriever, vision_retriever]
    if ocr_retriever is not None:
        retrievers.append(ocr_retriever)

    pipeline = VideoRAGPipeline(
        retrievers=retrievers,
        reranker=reranker,
        generator=QwenVLEvidenceGenerator(
            service,
            max_images=config.generation.max_images,
            evidence_mode=config.generation.evidence_mode,
            max_frames=config.generation.max_frames,
            video_fps=config.generation.video_fps,
            unload_after_generate=low_vram,
        ),
        recall_top_k={
            sparse_retriever.name: config.retrieval.sparse_top_k,
            "text_dense": config.retrieval.text_top_k,
            vision_retriever.name: config.retrieval.vision_top_k,
            **(
                {ocr_retriever.name: config.retrieval.ocr_top_k}
                if ocr_retriever is not None
                else {}
            ),
        },
        fusion_top_k=config.retrieval.fusion_top_k,
        rerank_top_k=config.retrieval.rerank_top_k,
        rrf_k=config.retrieval.rrf_k,
        minimum_rerank_score=config.generation.minimum_rerank_score,
        minimum_generator_confidence=config.generation.minimum_generator_confidence,
        allow_abstention=config.generation.allow_abstention,
        require_citations=config.generation.require_citations,
        retrieval_strategy=config.retrieval.strategy,
        primary_minimum_scores={
            sparse_retriever.name: config.retrieval.sparse_primary_min_score,
            "text_dense": config.retrieval.text_primary_min_score,
            vision_retriever.name: config.retrieval.vision_primary_min_score,
            **(
                {ocr_retriever.name: config.retrieval.ocr_primary_min_score}
                if ocr_retriever is not None
                else {}
            ),
        },
        minimum_route_confidence=config.retrieval.minimum_route_confidence,
        minimum_primary_score_margin=config.retrieval.minimum_primary_score_margin,
        fusion_policy=(
            AdaptiveFusionPolicy(
                sparse_source=sparse_retriever.name,
                text_source="text_dense",
                vision_source=vision_retriever.name,
                ocr_source=ocr_retriever.name if ocr_retriever is not None else None,
                sparse_weight=config.retrieval.sparse_weight,
                text_weight=config.retrieval.text_weight,
                vision_weight=config.retrieval.vision_weight,
                ocr_weight=config.retrieval.ocr_weight,
                visual_sparse_weight=config.retrieval.visual_sparse_weight,
                visual_text_weight=config.retrieval.visual_text_weight,
                visual_vision_weight=config.retrieval.visual_vision_weight,
                visual_ocr_weight=config.retrieval.visual_ocr_weight,
                ocr_query_sparse_weight=config.retrieval.ocr_query_sparse_weight,
                ocr_query_text_weight=config.retrieval.ocr_query_text_weight,
                ocr_query_vision_weight=config.retrieval.ocr_query_vision_weight,
                ocr_query_ocr_weight=config.retrieval.ocr_query_ocr_weight,
                agreement_bonus=config.retrieval.agreement_bonus,
            )
            if (
                config.retrieval.adaptive_fusion
                or config.retrieval.strategy == "cascade"
            )
            else None
        ),
        reranker_weight=config.retrieval.reranker_weight,
        neighbor_hops=config.retrieval.neighbor_hops,
        temporal_neighbor_hops=config.retrieval.temporal_neighbor_hops,
        dedupe_overlap_ratio=config.retrieval.dedupe_overlap_ratio,
        max_generation_segments=config.retrieval.max_generation_segments,
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
    print("[Warmup] Loading retrieval models...")
    pipeline.warmup()
    print("[Warmup] Retrieval models are ready.")
    create_app(pipeline).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
