import pytest

from video_rag.retrieval import BM25Retriever
from video_rag.schemas import VideoSegment


def segment(identifier: str, transcript: str) -> VideoSegment:
    return VideoSegment(identifier, "video", "video.mp4", 0, 10, transcript=transcript)


def test_bm25_applies_document_length_normalization():
    retriever = BM25Retriever(k1=1.2, b=0.75)
    retriever.build(
        [
            segment("short", "目标词"),
            segment("long", "目标词 " + "无关内容 " * 30),
        ]
    )

    results = retriever.search("目标词", 2)

    assert [result.segment_id for result in results] == ["short", "long"]
    assert results[0].score > results[1].score
    assert all(result.source == "bm25" for result in results)


@pytest.mark.parametrize("kwargs", [{"k1": 0}, {"b": -0.1}, {"b": 1.1}])
def test_bm25_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        BM25Retriever(**kwargs)

