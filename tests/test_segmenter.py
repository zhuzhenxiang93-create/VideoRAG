import unittest

from video_rag.ingestion import build_semantic_windows, build_windows, materialize_segments
from video_rag.schemas import Keyframe, OCRText, TimedText


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

    def test_semantic_windows_prefer_asr_boundaries(self):
        windows = build_semantic_windows(
            55,
            [TimedText(0, 18, "第一段"), TimedText(18, 37, "第二段")],
            target_seconds=20,
            overlap_seconds=4,
            minimum_seconds=8,
            maximum_seconds=28,
        )

        self.assertEqual(windows[0], (0.0, 18))
        self.assertEqual(windows[-1][1], 55)

    def test_materialize_segments_aligns_ocr(self):
        segments = materialize_segments(
            video_id="v1",
            source_path="v1.mp4",
            duration=20,
            transcript=[],
            ocr_items=[OCRText(4, "新西兰 2026", 0.9)],
        )

        self.assertEqual(segments[0].ocr_text, "新西兰 2026")
        self.assertIn("[OCR] 新西兰 2026", segments[0].evidence_text)
