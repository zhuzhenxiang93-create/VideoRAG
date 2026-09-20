import unittest

from video_rag.evaluation.tutorialvqa import (
    best_temporal_iou,
    interval_iou,
    latency_summary,
    relevant_segment_ids,
)
from video_rag.schemas import VideoSegment


class TutorialVQAEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            VideoSegment("v1_0", "v1", "v1.mp4", 0, 20),
            VideoSegment("v1_1", "v1", "v1.mp4", 15, 35),
            VideoSegment("v1_2", "v1", "v1.mp4", 30, 50),
            VideoSegment("v2_0", "v2", "v2.mp4", 0, 20),
        ]

    def test_relevant_segments_use_video_scope_and_temporal_overlap(self):
        result = relevant_segment_ids(
            self.segments, "v1", [{"start_time": 18, "end_time": 32}]
        )
        self.assertEqual(result, ["v1_0", "v1_1", "v1_2"])

    def test_interval_iou_and_best_iou(self):
        self.assertAlmostEqual(interval_iou(0, 20, 10, 30), 1 / 3)
        self.assertAlmostEqual(
            best_temporal_iou(self.segments[:2], [{"start_time": 15, "end_time": 35}]),
            1.0,
        )

    def test_latency_summary(self):
        self.assertEqual(
            latency_summary([1.0, 2.0, 10.0]),
            {"mean": 13 / 3, "median": 2.0, "p95": 10.0},
        )


if __name__ == "__main__":
    unittest.main()
