from video_rag.evaluation.dataset_validation import validate_questions


SEGMENT = {
    "segment_id": "video_0001",
    "video_id": "video",
    "start_time": 0.0,
    "end_time": 20.0,
    "keyframes": [{"timestamp": 2.0, "path": "frame.jpg"}],
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


def test_verified_status_requires_append_only_human_review_provenance():
    report = validate_questions(
        [question(verification_status="verified", annotation_source="manual")], [SEGMENT]
    )
    assert not report.valid
    assert any("reviewer_id" in value for value in report.errors)
    assert any("review_event_id" in value for value in report.errors)


def test_valid_human_review_can_remain_unsplit_until_freeze():
    item = question(
        verification_status="verified",
        annotation_source="human_review",
        reviewer_id="project_owner",
        reviewed_at="2026-08-21T10:00:00+00:00",
        review_event_id="event-1",
        modality_evidence={"visual": {"frame_timestamps": [2.0], "human_observation": "红旗", "human_verified": True}},
    )
    assert validate_questions([item], [SEGMENT]).valid
    assert not validate_questions([item], [SEGMENT], require_verified_splits=True).valid


def test_same_video_cannot_leak_across_frozen_splits():
    common = {
        "verification_status": "verified",
        "annotation_source": "human_review",
        "reviewer_id": "project_owner",
        "reviewed_at": "2026-08-21T10:00:00+00:00",
        "modality_evidence": {"visual": {"frame_timestamps": [2.0], "human_observation": "红旗", "human_verified": True}},
    }
    first = question(**common, review_event_id="event-1", split="development")
    second = question(**common, question_id="q2", review_event_id="event-2", split="test")
    report = validate_questions([first, second], [SEGMENT], require_verified_splits=True)
    assert not report.valid
    assert any("leaks across splits" in value for value in report.errors)


def test_unknown_route_is_independent_from_answerable_false():
    item = question(
        question_type="unknown_route", answerable=False, answer="", answer_aliases=[],
        relevant_segment_ids=[], evidence_start=None, evidence_end=None,
    )
    assert validate_questions([item], [SEGMENT]).valid
    assert validate_questions([{**item, "question_type": "visual"}], [SEGMENT]).valid
    report = validate_questions([{**item, "question_type": "unknown"}], [SEGMENT])
    assert not report.valid


def test_verified_ocr_requires_text_and_frame_evidence():
    item = question(
        question_type="ocr", verification_status="verified", annotation_source="human_review",
        reviewer_id="owner", reviewed_at="2026-08-21T10:00:00+00:00", review_event_id="event-ocr",
        modality_evidence={"visual": {"frame_timestamps": [2.0]}},
    )
    report = validate_questions([item], [SEGMENT])
    assert not report.valid
    assert any("OCR text" in value for value in report.errors)


def test_verified_unanswerable_requires_non_overlapping_full_video_check():
    item = question(
        question_type="unknown_route", answerable=False, answer="", answer_aliases=[],
        relevant_segment_ids=[], evidence_start=None, evidence_end=None,
        verification_status="verified", annotation_source="human_review", reviewer_id="owner",
        reviewed_at="2026-08-21T10:00:00+00:00", review_event_id="event-u",
        modality_evidence={}, unanswerable_reason="entity_absent", checked_time_ranges=[[0.0, 20.0]],
    )
    assert validate_questions([item], [SEGMENT]).valid
    report = validate_questions([{**item, "checked_time_ranges": [[0.0, 12.0], [10.0, 20.0]]}], [SEGMENT])
    assert not report.valid
    assert any("overlap" in value for value in report.errors)
