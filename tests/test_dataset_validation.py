from video_rag.evaluation.dataset_validation import validate_questions


SEGMENT = {
    "segment_id": "video_0001",
    "video_id": "video",
    "start_time": 0.0,
    "end_time": 20.0,
}


def question(**updates):
    item = {
        "question_id": "q1",
        "question": "画面里有什么？",
        "answer": "红旗",
        "answer_aliases": ["红旗", "红色旗帜"],
        "question_type": "visual",
        "video_id": "video",
        "relevant_segment_ids": ["video_0001"],
        "evidence_start": 1.0,
        "evidence_end": 5.0,
        "answerable": True,
        "verification_status": "generated_candidate",
        "annotation_source": "generated from existing seed set; pending review",
    }
    item.update(updates)
    return item


def test_valid_candidate_is_accepted_but_not_verified():
    report = validate_questions([question()], [SEGMENT])
    assert report.valid
    assert report.verified_count == 0
    assert report.candidate_count == 1
    assert any("do not use for resume" in value for value in report.warnings)


def test_missing_required_metadata_is_rejected():
    item = question()
    del item["answerable"]
    report = validate_questions([item], [SEGMENT])
    assert not report.valid
    assert "answerable" in report.errors[0]


def test_unanswerable_question_cannot_leak_evidence():
    report = validate_questions(
        [question(answerable=False, answer="", answer_aliases=[], evidence_start=None, evidence_end=None)],
        [SEGMENT],
    )
    assert not report.valid
    assert any("must not contain relevant segments" in value for value in report.errors)
