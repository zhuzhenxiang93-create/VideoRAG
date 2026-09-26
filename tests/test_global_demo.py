from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval.routing import AdaptiveFusionPolicy
from video_rag.schemas import GeneratedAnswer, SearchHit, VideoSegment


class StaticRetriever:
    def __init__(self, name, ranked):
        self.name = name
        self.ranked = ranked
        self.calls = []

    def build(self, segments):
        pass

    def search(self, query, top_k, *, allowed_ids=None):
        self.calls.append(allowed_ids)
        return [
            SearchHit(sid, 1 / rank, self.name, rank)
            for rank, sid in enumerate(self.ranked, 1)
            if allowed_ids is None or sid in allowed_ids
        ][:top_k]


class Reranker:
    supports_confidence = False

    def score(self, query, segments):
        return [1.0 if segment.segment_id == "b" else 0.1 for segment in segments]


class Generator:
    def generate(self, query, segments):
        return GeneratedAnswer("answer", True, (segments[0].segment_id,), 0.9)


def test_global_hybrid_uses_both_retrievers_and_reranks():
    lexical = StaticRetriever("bm25", ["a"])
    semantic = StaticRetriever("text_dense", ["b"])
    policy = AdaptiveFusionPolicy("bm25", "text_dense", "vision_dense")
    pipeline = VideoRAGPipeline(
        retrievers=[lexical, semantic], reranker=Reranker(), generator=Generator(),
        retrieval_strategy="hybrid", fusion_policy=policy,
        recall_top_k=2, fusion_top_k=2, rerank_top_k=2,
        reranker_weight=1.0, dedupe_overlap_ratio=0,
    )
    pipeline.build([
        VideoSegment("a", "A", "/a.webm", 0, 10, transcript="first"),
        VideoSegment("b", "B", "/b.webm", 0, 10, transcript="second"),
    ])
    assert [item.segment.segment_id for item in pipeline.search("question")] == ["a", "b"]
    assert [item.segment.segment_id for item in pipeline.search("question", rerank=True)] == ["b", "a"]
    assert pipeline.ask("question").citations == ("b",)
    assert lexical.calls == [None, None, None]
    assert semantic.calls == [None, None, None]
    assert [item.segment.segment_id for item in pipeline.search("question", ["A"])] == ["a"]


def test_global_demo_api_defaults_to_all_videos():
    from video_rag.api import create_app

    lexical = StaticRetriever("bm25", ["a"])
    semantic = StaticRetriever("text_dense", ["b"])
    pipeline = VideoRAGPipeline(
        retrievers=[lexical, semantic], reranker=Reranker(), generator=Generator(),
        retrieval_strategy="hybrid", recall_top_k=2, fusion_top_k=2,
        rerank_top_k=2, reranker_weight=1.0, dedupe_overlap_ratio=0,
    )
    pipeline.build([
        VideoSegment("a", "A", "/a.webm", 0, 10, transcript="first"),
        VideoSegment("b", "B", "/b.webm", 0, 10, transcript="second"),
    ])
    client = create_app(pipeline).test_client()
    page = client.get("/")
    assert page.status_code == 200 and "搜索全部视频" in page.get_data(as_text=True)
    found = client.post("/api/search", json={"question": "question", "rerank": True})
    assert found.status_code == 200 and found.json["evidence"][0]["video_id"] == "B"
    answer = client.post("/api/ask", json={"question": "question"})
    assert answer.status_code == 200 and answer.json["evidence"][0]["video_id"] == "B"
    scoped = client.post("/api/search", json={"question": "question", "video_ids": ["A"]})
    assert scoped.json["evidence"][0]["video_id"] == "A"
