import unittest

from video_rag.adapters import EvidenceGenerator, TokenOverlapReranker
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval import InMemoryLexicalRetriever
from video_rag.schemas import VideoSegment


def make_pipeline(minimum_score=0.01):
    pipeline = VideoRAGPipeline(
        retrievers=[InMemoryLexicalRetriever()],
        reranker=TokenOverlapReranker(),
        generator=EvidenceGenerator(),
        minimum_rerank_score=minimum_score,
    )
    pipeline.build(
        [
            VideoSegment(
                segment_id="v1_0000",
                video_id="v1",
                source_path="v1.mp4",
                start_time=0,
                end_time=20,
                transcript="飞机在机场附近降落",
            ),
            VideoSegment(
                segment_id="v2_0000",
                video_id="v2",
                source_path="v2.mp4",
                start_time=0,
                end_time=20,
                transcript="厨师正在制作蛋糕",
            ),
        ]
    )
    return pipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_returns_structured_evidence(self):
        result = make_pipeline().ask("飞机在哪里降落")
        self.assertFalse(result.abstained)
        self.assertEqual(result.evidence[0].segment.segment_id, "v1_0000")
        self.assertIn("v1_0000", result.answer)
        self.assertGreaterEqual(result.latency_ms["total"], 0)

    def test_pipeline_abstains_for_unrelated_question(self):
        result = make_pipeline().ask("量子计算机")
        self.assertTrue(result.abstained)
        self.assertEqual(result.evidence, ())
