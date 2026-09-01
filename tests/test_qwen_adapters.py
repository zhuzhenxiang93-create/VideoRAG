import unittest

from video_rag.adapters.qwen import QwenVLEvidenceGenerator, parse_generated_answer
from video_rag.schemas import VideoSegment


class FakeService:
    max_pixels = 1024

    def __init__(self):
        self.unloaded = False

    def infer(self, content):
        return "generated"

    def unload(self):
        self.unloaded = True


class QwenAdapterTests(unittest.TestCase):
    def test_low_vram_generator_unloads_service_after_generation(self):
        service = FakeService()
        generator = QwenVLEvidenceGenerator(service, unload_after_generate=True)
        segment = VideoSegment(
            segment_id="s1",
            video_id="v1",
            source_path="video.mp4",
            start_time=0,
            end_time=10,
            transcript="evidence",
        )
        result = generator.generate("question", [segment])
        self.assertEqual(result.answer, "generated")
        self.assertTrue(service.unloaded)

    def test_structured_generation_parser_preserves_citations(self):
        result = parse_generated_answer(
            '{"answerable":true,"answer":"机场","confidence":0.8,'
            '"citations":["s1"]}',
            {"s1"},
        )

        self.assertTrue(result.answerable)
        self.assertEqual(result.citations, ("s1",))
        self.assertEqual(result.confidence, 0.8)
