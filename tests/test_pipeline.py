import unittest

from video_rag.adapters import EvidenceGenerator, TokenOverlapReranker
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval import InMemoryLexicalRetriever
from video_rag.schemas import SearchHit, VideoSegment


class RecordingRetriever:
    def __init__(self, name):
        self.name = name
        self.top_k = None
        self.segment_id = None

    def build(self, segments):
        self.segment_id = segments[0].segment_id

    def search(self, query, top_k):
        self.top_k = top_k
        return [SearchHit(self.segment_id, 1.0, self.name, 1)]


def make_pipeline(minimum_score=0.01, allow_abstention=True):
    pipeline = VideoRAGPipeline(
        retrievers=[InMemoryLexicalRetriever()],
        reranker=TokenOverlapReranker(),
        generator=EvidenceGenerator(),
        minimum_rerank_score=minimum_score,
        allow_abstention=allow_abstention,
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

    def test_pipeline_uses_independent_recall_limits(self):
        first = RecordingRetriever("first")
        second = RecordingRetriever("second")
        pipeline = VideoRAGPipeline(
            retrievers=[first, second],
            reranker=TokenOverlapReranker(),
            generator=EvidenceGenerator(),
            recall_top_k={"first": 3, "second": 7},
            minimum_rerank_score=0,
        )
        pipeline.build(
            [VideoSegment("s1", "v1", "v1.mp4", 0, 10, transcript="证据")]
        )

        pipeline.ask("证据")

        self.assertEqual(first.top_k, 3)
        self.assertEqual(second.top_k, 7)

    def test_allow_abstention_false_bypasses_low_score_threshold(self):
        pipeline = make_pipeline(minimum_score=2.0, allow_abstention=False)

        result = pipeline.ask("飞机在哪里降落")

        self.assertFalse(result.abstained)
