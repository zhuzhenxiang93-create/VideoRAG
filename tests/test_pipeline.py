import unittest

from video_rag.adapters import EvidenceGenerator, FusionOrderReranker, TokenOverlapReranker
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval import InMemoryLexicalRetriever
from video_rag.retrieval.routing import AdaptiveFusionPolicy
from video_rag.schemas import GeneratedAnswer, SearchHit, VideoSegment


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


class StaticRetriever:
    name = "static"

    def __init__(self, identifiers):
        self.identifiers = identifiers

    def build(self, segments):
        del segments

    def search(self, query, top_k):
        del query
        return [
            SearchHit(identifier, 1.0 / rank, self.name, rank)
            for rank, identifier in enumerate(self.identifiers[:top_k], start=1)
        ]


class CapturingGenerator:
    def __init__(self, result=None):
        self.segment_ids = []
        self.result = result

    def generate(self, query, segments):
        del query
        self.segment_ids = [segment.segment_id for segment in segments]
        return self.result or GeneratedAnswer(
            "有证据的答案", True, citations=(segments[0].segment_id,), confidence=0.9
        )


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

    def test_adaptive_policy_skips_disabled_retrieval_routes(self):
        sparse = RecordingRetriever("sparse")
        text = RecordingRetriever("text")
        vision = RecordingRetriever("vision")
        policy = AdaptiveFusionPolicy(
            sparse_source="sparse",
            text_source="text",
            vision_source="vision",
        )
        pipeline = VideoRAGPipeline(
            retrievers=[sparse, text, vision],
            reranker=TokenOverlapReranker(),
            generator=EvidenceGenerator(),
            fusion_policy=policy,
            minimum_rerank_score=0,
        )
        pipeline.build(
            [VideoSegment("s1", "v1", "v1.mp4", 0, 10, transcript="选举原则")]
        )

        pipeline.ask("要落实什么原则？")

        self.assertEqual(sparse.top_k, 20)
        self.assertIsNone(text.top_k)
        self.assertIsNone(vision.top_k)

    def test_overlapping_retrieval_candidates_are_deduplicated(self):
        generator = CapturingGenerator()
        pipeline = VideoRAGPipeline(
            retrievers=[StaticRetriever(["s1", "s2"])],
            reranker=FusionOrderReranker(),
            generator=generator,
            neighbor_hops=0,
            dedupe_overlap_ratio=0.2,
        )
        pipeline.build(
            [
                VideoSegment("s1", "v1", "v1.mp4", 0, 20, transcript="证据"),
                VideoSegment("s2", "v1", "v1.mp4", 15, 35, transcript="重复证据"),
            ]
        )

        pipeline.ask("证据")

        self.assertEqual(generator.segment_ids, ["s1"])

    def test_temporal_route_expands_neighbor_context(self):
        generator = CapturingGenerator()
        policy = AdaptiveFusionPolicy("static", "text", "vision")
        pipeline = VideoRAGPipeline(
            retrievers=[StaticRetriever(["s2"])],
            reranker=FusionOrderReranker(),
            generator=generator,
            fusion_policy=policy,
            neighbor_hops=0,
            temporal_neighbor_hops=1,
            dedupe_overlap_ratio=0,
        )
        pipeline.build(
            [
                VideoSegment("s1", "v1", "v1.mp4", 0, 10, transcript="开始"),
                VideoSegment("s2", "v1", "v1.mp4", 10, 20, transcript="中间"),
                VideoSegment("s3", "v1", "v1.mp4", 20, 30, transcript="结束"),
            ]
        )

        result = pipeline.ask("随后发生了什么？")

        self.assertEqual(generator.segment_ids, ["s1", "s2", "s3"])
        self.assertIn("temporal", result.route_labels)

    def test_invalid_generated_citation_forces_abstention(self):
        generator = CapturingGenerator(
            GeneratedAnswer("错误引用", True, citations=("missing",), confidence=0.9)
        )
        pipeline = VideoRAGPipeline(
            retrievers=[StaticRetriever(["s1"])],
            reranker=FusionOrderReranker(),
            generator=generator,
        )
        pipeline.build(
            [VideoSegment("s1", "v1", "v1.mp4", 0, 10, transcript="证据")]
        )

        result = pipeline.ask("问题")

        self.assertTrue(result.abstained)
        self.assertEqual(result.citations, ())

    def test_fusion_order_is_not_treated_as_calibrated_confidence(self):
        pipeline = VideoRAGPipeline(
            retrievers=[StaticRetriever(["s1"])],
            reranker=FusionOrderReranker(),
            generator=CapturingGenerator(),
            minimum_rerank_score=2.0,
        )
        pipeline.build(
            [VideoSegment("s1", "v1", "v1.mp4", 0, 10, transcript="证据")]
        )

        self.assertFalse(pipeline.ask("证据").abstained)
