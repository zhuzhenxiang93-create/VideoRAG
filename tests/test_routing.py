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


def test_ocr_question_activates_independent_ocr_route():
    routed = AdaptiveFusionPolicy(
        sparse_source="bm25",
        text_source="text",
        vision_source="vision",
        ocr_source="ocr",
    )

    decision = routed.decision("屏幕右下角显示的数字是多少？")

    assert decision.labels == ("ocr",)
    assert decision.source_weights["ocr"] > decision.source_weights["vision"]


def test_multimodal_temporal_query_keeps_multiple_labels():
    decision = policy().decision("结合画面和解说，随后发生了什么？")

    assert set(decision.labels) == {"visual", "multimodal", "temporal"}


def test_exact_fact_uses_sparse_then_text_cascade():
    decision = policy().decision("完善选举制度要落实什么原则？")

    assert decision.primary_sources == ("bm25",)
    assert decision.fallback_sources == ("text",)
    assert decision.candidate_mode == "cascade"


def test_semantic_summary_uses_dense_then_sparse_cascade():
    decision = policy().decision("请总结这段视频的主要内容")

    assert "semantic" in decision.labels
    assert decision.primary_sources == ("text",)
    assert decision.fallback_sources == ("bm25",)


def test_multimodal_route_uses_candidate_union_without_ocr_by_default():
    decision = policy().decision("结合画面和解说，这段内容表达了什么？")

    assert decision.primary_sources == ("bm25", "text", "vision")
    assert decision.fallback_sources == ()
    assert decision.candidate_mode == "union"


def test_multimodal_route_does_not_enable_ocr_without_ocr_intent():
    routed = AdaptiveFusionPolicy(
        sparse_source="bm25",
        text_source="text",
        vision_source="vision",
        ocr_source="ocr",
    )

    decision = routed.decision("结合画面和解说，这段内容表达了什么？")

    assert decision.source_weights["ocr"] == 0.0
    assert "ocr" not in decision.primary_sources
