import numpy as np

from video_rag.retrieval.frame_faiss import FrameClipVisionRetriever, aggregate_frame_hits, collect_physical_frames
from video_rag.schemas import Keyframe, VideoSegment


def segment(segment_id: str, start: float, frame: Keyframe) -> VideoSegment:
    return VideoSegment(
        segment_id=segment_id,
        video_id="video",
        source_path="video.mp4",
        start_time=start,
        end_time=start + 20,
        keyframes=(frame,),
    )


def test_collect_physical_frames_encodes_shared_path_once():
    shared = Keyframe(timestamp=17.0, path="frames/shared.jpg")
    frames = collect_physical_frames([segment("s0", 0, shared), segment("s1", 15, shared)])
    assert len(frames) == 1
    assert frames[0]["segment_ids"] == ["s0", "s1"]
    assert frames[0]["frame_path"] == "frames/shared.jpg"


def test_max_gives_shared_frame_score_to_both_segments_and_stable_ties():
    frame = {"frame_id": "f", "segment_ids": ["s1", "s0"], "timestamp": 17.0}
    assert aggregate_frame_hits([(frame, 0.8)], "max") == [("s0", 0.8), ("s1", 0.8)]


def test_top2_mean_uses_one_score_for_single_frame_and_best_two_otherwise():
    hits = [
        ({"segment_ids": ["single", "multi"]}, 0.9),
        ({"segment_ids": ["multi"]}, 0.7),
        ({"segment_ids": ["multi"]}, 0.1),
    ]
    assert aggregate_frame_hits(hits, "top2_mean") == [("single", 0.9), ("multi", 0.8)]


def test_invalid_aggregation_fails_closed():
    try:
        aggregate_frame_hits([], "mean_all")
    except ValueError as error:
        assert "max" in str(error)
    else:
        raise AssertionError("invalid aggregation must fail")


def test_builder_encodes_unique_frames_not_memberships(tmp_path):
    image = tmp_path / "shared.jpg"
    image.write_bytes(b"test fixture; encoder is mocked")
    shared = Keyframe(timestamp=17.0, path=str(image))
    segments = [segment("s0", 0, shared), segment("s1", 15, shared)]
    encoded_counts = []

    class FakeFrameRetriever(FrameClipVisionRetriever):
        def _encode_images(self, frames):
            encoded_counts.append(len(frames))
            return np.ones((len(frames), 4), dtype=np.float32) / 2

    retriever = FakeFrameRetriever("fake/chinese-clip", index_dir=tmp_path, force_rebuild=True)
    retriever.build(segments)
    assert encoded_counts == [1]
    assert retriever._index.ntotal == 1
    assert retriever._frames[0]["segment_ids"] == ["s0", "s1"]


def test_search_returns_best_frame_timestamp_as_evidence(tmp_path):
    import faiss

    class FakeQueryRetriever(FrameClipVisionRetriever):
        def _encode_query(self, query):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    retriever = FakeQueryRetriever("fake/chinese-clip", index_dir=tmp_path, frame_candidate_k=2)
    retriever._index = faiss.IndexFlatIP(2)
    retriever._index.add(np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32))
    retriever._frames = [
        {"frame_id": "best", "frame_path": "best.jpg", "timestamp": 12.0, "segment_ids": ["s"]},
        {"frame_id": "other", "frame_path": "other.jpg", "timestamp": 18.0, "segment_ids": ["s"]},
    ]
    hit = retriever.search("query", 1)[0]
    assert hit.metadata["best_frame_id"] == "best"
    assert hit.metadata["best_frame_timestamp"] == 12.0
