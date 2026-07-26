import unittest

from video_rag.ingestion import build_windows, materialize_segments
from video_rag.schemas import Keyframe, TimedText


class SegmenterTests(unittest.TestCase):
    def test_build_windows_covers_video_with_overlap(self):
        self.assertEqual(
            build_windows(42, 20, 5),
            [(0.0, 20.0), (15.0, 35.0), (30.0, 42)],
        )

    def test_materialize_segments_aligns_transcript_and_keyframes(self):
        segments = materialize_segments(
            video_id="v1",
            source_path="v1.mp4",
            duration=30,
            transcript=[
                TimedText(2, 4, "第一句"),
                TimedText(17, 22, "跨窗口句子"),
            ],
            keyframes=[Keyframe(18, "frame.jpg", "飞机画面")],
            window_seconds=20,
            overlap_seconds=5,
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].transcript, "第一句 跨窗口句子")
        self.assertEqual(segments[1].transcript, "跨窗口句子")
        self.assertEqual(segments[0].visual_caption, "飞机画面")
        self.assertEqual(segments[1].visual_caption, "飞机画面")
