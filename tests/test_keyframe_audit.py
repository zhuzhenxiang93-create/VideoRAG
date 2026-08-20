from scripts.audit_keyframes import hamming, longest_zero_run, percentile, plan_supplement


def test_hamming_distance():
    assert hamming(0b0000, 0b0000) == 0
    assert hamming(0b0000, 0b1011) == 3


def test_percentile_linear_interpolation():
    assert percentile([], 0.5) is None
    assert percentile([0, 10], 0.5) == 5
    assert percentile([1, 2, 3], 0.9) == 2.8


def test_longest_zero_run_uses_temporal_order():
    segments = [
        {"segment_id": "b", "start_time": 15, "end_time": 35, "keyframes": []},
        {"segment_id": "a", "start_time": 0, "end_time": 20, "keyframes": []},
        {"segment_id": "c", "start_time": 30, "end_time": 50, "keyframes": [{"timestamp": 31}]},
    ]
    result = longest_zero_run(segments)
    assert result["count"] == 2
    assert result["span_seconds"] == 35
    assert result["segment_ids"] == ["a", "b"]


def test_supplement_plan_reuses_one_physical_frame_across_overlap():
    segments = [
        {"segment_id": "a", "start_time": 0, "end_time": 20, "keyframes": []},
        {"segment_id": "b", "start_time": 15, "end_time": 35, "keyframes": []},
    ]
    plan = plan_supplement(segments, 1)
    # One timestamp in [15, 20) is decoded once and shared by both windows.
    assert plan["new_physical_frames"] == 1
    assert plan["new_memberships"] == 2
    assert plan["overlap_reused_memberships"] == 1
    assert plan["clip_encodes_required"] == 1


def test_supplement_plan_can_share_candidate_in_overlap():
    segments = [
        {"segment_id": "a", "start_time": 0, "end_time": 20, "keyframes": [{"timestamp": 5}]},
        {"segment_id": "b", "start_time": 15, "end_time": 35, "keyframes": []},
    ]
    plan = plan_supplement(segments, 2)
    assert plan["new_physical_frames"] == 2
    assert plan["new_memberships"] == 3
    assert plan["overlap_reused_memberships"] == 1
