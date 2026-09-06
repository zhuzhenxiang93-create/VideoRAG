"""Preserve v1 artifacts and rebuild tutorial segments with bounded Whisper timestamps."""

import json
from dataclasses import asdict
from pathlib import Path

from video_rag.ingestion import WhisperTranscriber, materialize_segments, probe_video
from video_rag.schemas import TimedText
from video_rag.storage import load_segments, save_segments

root = Path("artifacts/tutorialvqa")
target = root / "bounded-v2"
target.mkdir(exist_ok=True)
original = load_segments(root / "segments.ocr.jsonl")
whisper = WhisperTranscriber()
result = []
catalog = []
for vid in sorted({s.video_id for s in original}):
    old = [s for s in original if s.video_id == vid]
    path = old[0].source_path
    info = probe_video(path)
    cached = target / f"{vid}.whisper.json"
    if cached.exists():
        transcript = [TimedText(**r) for r in json.loads(cached.read_text())]
    else:
        print("Bounded Whisper", vid, flush=True)
        transcript = whisper.transcribe_bounded(path, language="en", seconds=20)
        assert transcript
        cached.write_text(json.dumps([asdict(r) for r in transcript], ensure_ascii=False, indent=2))
    assert all(
        0 <= r.start_time < r.end_time <= info.duration + 0.1 and r.end_time - r.start_time <= 20.01
        for r in transcript
    )
    frames = {f.path: f for s in old for f in s.keyframes}
    ocr = {(i.timestamp, i.text, i.bbox): i for s in old for i in s.ocr_items}
    result.extend(
        materialize_segments(
            video_id=vid,
            source_path=path,
            duration=info.duration,
            transcript=transcript,
            keyframes=frames.values(),
            ocr_items=ocr.values(),
            strategy="fixed",
            window_seconds=20,
            overlap_seconds=5,
        )
    )
    catalog.append(
        {
            "video_id": vid,
            "duration": info.duration,
            "asr_chunks": len(transcript),
            "max_asr_span": max(r.end_time - r.start_time for r in transcript),
            "frames": len(frames),
            "ocr_items": len(ocr),
            "transcript_source": "Whisper small, independent 20-second audio blocks; no official transcript or QA labels used",
        }
    )
    print(catalog[-1], flush=True)
whisper.unload()
save_segments(target / "segments.ocr.jsonl", result)
(target / "input-manifest.json").write_text(json.dumps(catalog, indent=2))
