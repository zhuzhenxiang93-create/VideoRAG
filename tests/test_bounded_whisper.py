from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from video_rag.ingestion.media import WhisperTranscriber


def test_bounded_whisper_offsets_and_clamps_to_real_audio():
    t = WhisperTranscriber(device="cpu")
    calls = []

    def fake(data, **kwargs):
        calls.append(len(data["array"]))
        return {
            "chunks": [
                {"timestamp": (0, 999), "text": "bounded evidence"},
                {"timestamp": (21, 22), "text": "past audio"},
            ]
        }

    t._pipeline = fake
    with patch(
        "subprocess.run",
        return_value=SimpleNamespace(stdout=np.zeros(25 * 16000, dtype="<f4").tobytes()),
    ):
        rows = t.transcribe_bounded("video.webm", language="en")
    assert calls == [20 * 16000, 5 * 16000]
    assert [(r.start_time, r.end_time) for r in rows] == [(0, 20), (20, 25)]


def test_bounded_whisper_rejects_invalid_block_size():
    with pytest.raises(ValueError):
        WhisperTranscriber().transcribe_bounded("video", seconds=0)
