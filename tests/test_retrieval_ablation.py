import json

import pytest

from scripts.evaluate_retrieval import load_questions, percentile


def test_percentile_handles_small_samples():
    assert percentile([], 0.95) == 0.0
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([1.0, 2.0, 10.0], 0.95) == 10.0


def test_load_questions_validates_unknown_segments(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "问题",
                "answer": "答案",
                "relevant_segment_ids": ["missing"],
                "question_type": "audio",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown segments"):
        load_questions(path, {"known"})
