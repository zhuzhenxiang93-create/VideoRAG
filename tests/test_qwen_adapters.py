import unittest

from video_rag.adapters.qwen import QwenVLEvidenceGenerator
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
        self.assertEqual(generator.generate("question", [segment]), "generated")
        self.assertTrue(service.unloaded)
