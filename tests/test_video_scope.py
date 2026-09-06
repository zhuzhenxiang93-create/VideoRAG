import numpy as np
import pytest

from video_rag.adapters import FusionOrderReranker
from video_rag.adapters.qwen import QwenVLEvidenceGenerator
from video_rag.api import create_app
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval.faiss_dense import FaissDenseRetriever
from video_rag.retrieval.in_memory import BM25Retriever
from video_rag.retrieval.routing import AdaptiveFusionPolicy
from video_rag.schemas import GeneratedAnswer, Keyframe, VideoSegment


def segment(sid, vid, text, frames=()):
    return VideoSegment(sid, vid, "/unavailable.mp4", 0, 10, transcript=text, keyframes=frames)


class Generator:
    def generate(self, query, segments):
        return GeneratedAnswer("supported", True, (segments[0].segment_id,), 0.9)


def pipeline(generator=None):
    p = VideoRAGPipeline(
        retrievers=[BM25Retriever()],
        reranker=FusionOrderReranker(),
        generator=generator or Generator(),
        retrieval_strategy="cascade",
        recall_top_k=1,
        fusion_top_k=1,
    )
    p.build([segment("b", "B", "button " * 20), segment("a", "A", "button")])
    return p


def test_scope_before_top_k_and_citations():
    p = pipeline()
    assert [e.segment.video_id for e in p.search("button", ["A"])] == ["A"]
    assert [e.segment.video_id for e in p.ask("button", ["A"]).evidence] == ["A"]
    for scope in [[], ["unknown"], "A"]:
        with pytest.raises(ValueError):
            p.ask("button", scope)


def test_api_scope_search_does_not_generate_and_validation():
    class Fail:
        def generate(self, *args):
            raise RuntimeError("generation failed")

    client = create_app(pipeline(Fail())).test_client()
    assert client.get("/api/videos").status_code == 200
    result = client.post("/api/search", json={"question": "button", "video_ids": ["A"]})
    assert result.json["evidence"][0]["video_id"] == "A"
    assert (
        client.post("/api/ask", json={"question": "button", "video_ids": ["A"]}).status_code == 503
    )
    assert client.post("/api/ask", json={"question": "button", "video_ids": []}).status_code == 400


def test_dense_scope_before_top_k():
    class Dense(FaissDenseRetriever):
        def encode_documents(self, segments):
            return np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32), ["b", "a"]

        def encode_query(self, query):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    d = Dense()
    d.build([segment("b", "B", ""), segment("a", "A", "")])
    assert d.search("q", 1)[0].segment_id == "b"
    assert d.search("q", 1, allowed_ids={"a"})[0].segment_id == "a"


def test_physical_frame_dedupe_and_segment_coverage(tmp_path):
    f1 = tmp_path / "one.jpg"
    f1.touch()
    f2 = tmp_path / "two.jpg"
    f2.touch()

    class Service:
        max_pixels = 1024

        def infer(self, content):
            self.content = content
            return '{"answerable":false,"answer":"insufficient","citations":[]}'

    service = Service()
    generator = QwenVLEvidenceGenerator(service, max_images=2)
    frames = (Keyframe(1, str(f1)), Keyframe(2, str(f2)))
    generator.generate(
        "q",
        [
            segment("a", "A", "", frames),
            segment("b", "A", "", frames[:1]),
            segment("c", "B", "", (Keyframe(3, str(f2)),)),
        ],
    )
    images = [c["image"] for c in service.content if c["type"] == "image"]
    assert images == [str(f1), str(f2)]
    assert any("time=3.000s" in c.get("text", "") for c in service.content)


def test_invalid_or_empty_generated_answer_is_not_success():
    class Bad:
        def generate(self, *args):
            return GeneratedAnswer("", True, ("a",), 0.9)

    assert pipeline(Bad()).ask("button", ["A"]).abstained

    class Outside:
        def generate(self, *args):
            return GeneratedAnswer("wrong", True, ("b",), 0.9)

    assert pipeline(Outside()).ask("button", ["A"]).abstained


def test_english_routing():
    p = AdaptiveFusionPolicy(
        sparse_source="bm25", text_source="text", vision_source="vision", ocr_source="ocr"
    )
    assert "ocr" in p.intents("Which button sets the parameter value?")
    assert "temporal" in p.intents("What happens next?")


def test_fallback_reserves_useful_primary_candidates():
    from video_rag.retrieval.routing import RoutingDecision
    from video_rag.schemas import SearchHit

    class Static:
        def __init__(self, name, ids):
            self.name, self.ids = name, ids

        def search(self, query, count):
            return [
                SearchHit(sid, 0.1, self.name, rank) for rank, sid in enumerate(self.ids[:count], 1)
            ]

    p = VideoRAGPipeline(
        retrievers=[Static("primary", ["p1", "p2"]), Static("fallback", ["f1", "f2"])],
        reranker=FusionOrderReranker(),
        generator=Generator(),
        retrieval_strategy="cascade",
        fusion_top_k=2,
        primary_minimum_scores={"primary": 0.5},
    )
    decision = RoutingDecision(("text",), {}, 0.9, ("primary",), ("fallback",), "cascade")
    assert [h.segment_id for h in p._retrieve("q", decision)] == ["f1", "p1"]
