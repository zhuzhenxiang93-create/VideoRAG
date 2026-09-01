from pathlib import Path

from scripts.enrich_ocr import enrich_segments
from video_rag.ingestion.ocr import PaddleOCRExtractor, deduplicate_ocr
from video_rag.retrieval import OCRBM25Retriever
from video_rag.schemas import Keyframe, OCRText, VideoSegment


class FakePaddleEngine:
    def ocr(self, path, cls=True):
        del path, cls
        return [
            [
                [[[0, 0], [10, 0], [10, 4], [0, 4]], ("增长 12%", 0.93)],
                [[[0, 5], [10, 5], [10, 9], [0, 9]], ("噪声", 0.20)],
            ]
        ]


def test_paddle_ocr_extracts_timestamped_high_confidence_text(tmp_path: Path):
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"frame")
    extractor = PaddleOCRExtractor(
        minimum_confidence=0.5,
        engine_factory=lambda **kwargs: FakePaddleEngine(),
    )

    items = extractor.extract([Keyframe(3.5, str(frame_path))])

    assert [(item.timestamp, item.text) for item in items] == [(3.5, "增长 12%")]
    assert items[0].bbox[2] == (10.0, 4.0)


def test_ocr_temporal_deduplication_keeps_stronger_observation():
    items = deduplicate_ocr(
        [OCRText(1, "直播", 0.7), OCRText(2, "直播", 0.95), OCRText(8, "直播", 0.8)]
    )

    assert [(item.timestamp, item.confidence) for item in items] == [(2, 0.95), (8, 0.8)]


def test_independent_ocr_retriever_does_not_search_asr():
    retriever = OCRBM25Retriever()
    retriever.build(
        [
            VideoSegment("s1", "v1", "v1.mp4", 0, 10, transcript="增长 12%"),
            VideoSegment("s2", "v1", "v1.mp4", 10, 20, ocr_text="增长 12%"),
        ]
    )

    assert [hit.segment_id for hit in retriever.search("增长 12%", 5)] == ["s2"]


def test_incremental_enrichment_reuses_physical_frames_across_overlaps():
    class RecordingExtractor:
        def __init__(self):
            self.frame_count = 0

        def extract(self, frames):
            self.frame_count += len(frames)
            return [OCRText(frame.timestamp, "直播", 0.9) for frame in frames]

    frame = Keyframe(17, "shared.jpg")
    segments = [
        VideoSegment("s1", "v1", "v1.mp4", 0, 20, keyframes=(frame,)),
        VideoSegment("s2", "v1", "v1.mp4", 15, 35, keyframes=(frame,)),
    ]
    extractor = RecordingExtractor()

    enriched = enrich_segments(segments, extractor)

    assert extractor.frame_count == 1
    assert [segment.ocr_text for segment in enriched] == ["直播", "直播"]


def test_paddle_v3_array_like_polygons_do_not_require_boolean_coercion(tmp_path: Path):
    class ArrayLike(list):
        def __bool__(self):
            raise ValueError("ambiguous truth value")

    class FakeV3Engine:
        def predict(self, path):
            del path
            polygon = ArrayLike(
                [
                    ArrayLike([0, 0]),
                    ArrayLike([10, 0]),
                    ArrayLike([10, 4]),
                    ArrayLike([0, 4]),
                ]
            )
            return [
                {
                    "res": {
                        "rec_texts": ["屏幕文字"],
                        "rec_scores": [0.91],
                        "dt_polys": ArrayLike([polygon]),
                    }
                }
            ]

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"frame")
    extractor = PaddleOCRExtractor(engine_factory=lambda **kwargs: FakeV3Engine())

    items = extractor.extract([Keyframe(2.0, str(frame_path))])

    assert items[0].text == "屏幕文字"
    assert items[0].bbox == ((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0))
