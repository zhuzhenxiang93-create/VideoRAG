from pathlib import Path

import numpy as np

from video_rag.adapters.qwen import QwenVLEvidenceGenerator
from video_rag.adapters.qwen3_vl import Qwen3VLReranker
from video_rag.retrieval.qwen3_vl import Qwen3VLEmbeddingRetriever
from video_rag.schemas import Keyframe, VideoSegment


class FakeOfficialModel:
    last_payload = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def process(self, payload):
        FakeOfficialModel.last_payload = payload
        if isinstance(payload, dict):
            return [0.8 for _ in payload["documents"]]
        return np.ones((len(payload), 4), dtype=np.float32)


class FakeGenerationService:
    max_pixels = 1024

    def __init__(self):
        self.content = None

    def infer(self, content):
        self.content = content
        return "answer"


def make_segment(frame_path: Path) -> VideoSegment:
    return VideoSegment(
        "s1",
        "v1",
        "video.mp4",
        0,
        10,
        transcript="飞机降落",
        keyframes=(Keyframe(1.0, str(frame_path)),),
    )


def test_qwen3_vl_embedding_builds_joint_text_image_inputs(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"test")
    retriever = Qwen3VLEmbeddingRetriever(
        implementation_repository=tmp_path,
        model_factory=FakeOfficialModel,
    )

    vectors, identifiers = retriever.encode_documents([make_segment(frame)])

    assert vectors.shape == (1, 4)
    assert identifiers == ["s1"]
    assert FakeOfficialModel.last_payload[0]["video"] == [str(frame)]
    assert "飞机降落" in FakeOfficialModel.last_payload[0]["text"]


def test_qwen3_vl_reranker_sends_multimodal_documents(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"test")
    reranker = Qwen3VLReranker(
        implementation_repository=tmp_path,
        model_factory=FakeOfficialModel,
    )

    scores = reranker.score("在哪里降落", [make_segment(frame)])

    assert scores == [0.8]
    assert FakeOfficialModel.last_payload["query"] == {"text": "在哪里降落"}
    assert FakeOfficialModel.last_payload["documents"][0]["video"] == [str(frame)]


def test_frame_sequence_generation_preserves_ordered_frames(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"test")
    service = FakeGenerationService()
    generator = QwenVLEvidenceGenerator(
        service,
        evidence_mode="frame_sequence",
        max_frames=8,
    )

    assert generator.generate("发生了什么", [make_segment(frame)]).answer == "answer"
    video_items = [item for item in service.content if item["type"] == "video"]
    assert video_items == [
        {"type": "video", "video": [str(frame)], "sample_fps": 1.0}
    ]
