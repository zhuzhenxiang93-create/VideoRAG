from __future__ import annotations

from video_rag.adapters import EvidenceGenerator, TokenOverlapReranker
from video_rag.api import create_app
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval import BM25Retriever
from video_rag.schemas import Keyframe, VideoSegment


def build_demo_pipeline() -> VideoRAGPipeline:
    segments = [
        VideoSegment(
            segment_id="demo_0000",
            video_id="demo",
            source_path="data/demo/demo.mp4",
            start_time=0,
            end_time=20,
            transcript="主持人介绍了多模态视频问答系统的整体架构。",
            visual_caption="画面展示召回、精排和生成三个模块。",
            keyframes=(Keyframe(8.0, "data/demo/frame_008.jpg", "系统架构图"),),
        ),
        VideoSegment(
            segment_id="demo_0001",
            video_id="demo",
            source_path="data/demo/demo.mp4",
            start_time=15,
            end_time=35,
            transcript="系统使用CLIP检索与问题相关的视频画面。",
            visual_caption="屏幕展示CLIP图像编码器和文本编码器。",
        ),
    ]
    pipeline = VideoRAGPipeline(
        retrievers=[BM25Retriever()],
        reranker=TokenOverlapReranker(),
        generator=EvidenceGenerator(),
        minimum_rerank_score=0.01,
    )
    pipeline.build(segments)
    return pipeline


if __name__ == "__main__":
    create_app(build_demo_pipeline()).run(host="127.0.0.1", port=5000, debug=False)
