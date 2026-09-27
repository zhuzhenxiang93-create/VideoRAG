from test_global_demo import Generator, Reranker, StaticRetriever

from video_rag.api import create_app
from video_rag.pipeline import VideoRAGPipeline
from video_rag.schemas import VideoSegment


def pipeline():
    retriever = StaticRetriever("bm25", ["a", "b"])
    result = VideoRAGPipeline(retrievers=[retriever], reranker=Reranker(),
                              generator=Generator(), rerank_top_k=2)
    result.build([VideoSegment("a", "A", "/a.webm", 0, 10),
                  VideoSegment("b", "B", "/b.webm", 0, 10)])
    return result, retriever


def test_preview_cannot_generate_and_has_real_metadata():
    model, _ = pipeline()
    client = create_app(model, video_titles={"A": "Real title"}, preview=True).test_client()
    assert client.get("/api/health").json["preview"]
    assert client.post("/api/ask", json={"question": "x"}).status_code == 503
    result = client.post("/api/search", json={"question": "x", "video_ids": ["A"]}).json
    assert result["evidence"][0]["video_title"] == "Real title"
    assert {e["video_id"] for e in result["evidence"]} == {"A"}
    assert client.post("/api/search", json=["invalid"]).status_code == 400


def test_frozen_pool_reranking_does_not_repeat_recall():
    model, retriever = pipeline()
    pool = model.search("x")
    calls = len(retriever.calls)
    ranked = model.rerank_candidates("x", pool)
    assert len(retriever.calls) == calls
    assert ranked[0].segment.segment_id == "b"
    assert {e.segment.segment_id for e in ranked} == {e.segment.segment_id for e in pool}
