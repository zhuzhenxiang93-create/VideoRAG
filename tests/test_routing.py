import pytest

from video_rag.retrieval.routing import AdaptiveFusionPolicy


def policy(**kwargs):
    return AdaptiveFusionPolicy(
        sparse_source="bm25",
        text_source="text",
        vision_source="vision",
        **kwargs,
    )


@pytest.mark.parametrize(
    "query",
    [
        "新闻画面中人们举着什么颜色的旗帜？",
        "图中人物穿着什么衣服？",
        "视频里出现了什么标志？",
    ],
)
def test_visual_intent_detection(query):
    assert policy().intent(query) == "visual"


def test_audio_fact_question_prefers_sparse_and_text_routes():
    weights = policy().source_weights("完善选举制度要落实什么原则？")

    assert weights == {"bm25": 1.0, "text": 0.0, "vision": 0.0}


def test_visual_question_prefers_vision_route():
    weights = policy().source_weights("画面中人们举着什么颜色的旗帜？")

    assert weights["vision"] > weights["bm25"] > weights["text"]
