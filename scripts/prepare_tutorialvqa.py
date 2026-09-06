"""Prepare five tutorials from real Whisper; never use annotated spans as inputs."""

import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import cv2

from video_rag.ingestion import WhisperTranscriber, materialize_segments, probe_video
from video_rag.schemas import Keyframe, TimedText
from video_rag.storage import save_segments

IDS = ["14643", "14644", "14645", "14646", "14647"]
ROOT = Path("artifacts/tutorialvqa")
ROOT.mkdir(parents=True, exist_ok=True)
raw = Path("data/tutorialvqa/videos")
raw.mkdir(parents=True, exist_ok=True)
missing = [v for v in IDS if not (raw / f"{v}.webm").exists()]
if missing:
    with zipfile.ZipFile("data/tutorialvqa/original/videos.zip") as archive:
        for vid in missing:
            (raw / f"{vid}.webm").write_bytes(archive.read(f"videos/{vid}.webm"))
whisper = WhisperTranscriber()
segments = []
catalog = []
for vid in IDS:
    path = (raw / f"{vid}.webm").resolve()
    info = probe_video(path)
    out = ROOT / vid
    out.mkdir(exist_ok=True)
    asr_path = out / "whisper.bounded.json"
    if asr_path.exists():
        transcript = [TimedText(**r) for r in json.loads(asr_path.read_text())]
    else:
        print("WHISPER", vid, flush=True)
        transcript = whisper.transcribe_bounded(path, language="en", seconds=20)
        assert transcript, f"Empty Whisper output for {vid}"
        asr_path.write_text(
            json.dumps([asdict(r) for r in transcript], ensure_ascii=False, indent=2)
        )
    # Screencasts may have few scene changes. Uniform 5-second samples provide
    # coverage independent of QA labels or annotated answer intervals.
    capture = cv2.VideoCapture(str(path))
    frames = []
    timestamp = 0.0
    while timestamp < info.duration:
        f = out / f"frame-{timestamp:08.3f}.jpg"
        if not f.exists():
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, image = capture.read()
            if not ok:
                timestamp += 5
                continue
            assert cv2.imwrite(str(f), image)
        frames.append(
            Keyframe(
                timestamp, str(f.resolve()), selection_source="uniform_5s", described_by_vlm=False
            )
        )
        timestamp += 5
    capture.release()
    assert frames
    segments.extend(
        materialize_segments(
            video_id=vid,
            source_path=str(path),
            duration=info.duration,
            transcript=transcript,
            keyframes=frames,
            strategy="fixed",
            window_seconds=20,
            overlap_seconds=5,
        )
    )
    catalog.append(
        {
            "video_id": vid,
            "duration": info.duration,
            "frames": len(frames),
            "asr_chunks": len(transcript),
            "transcript_source": "Whisper openai/whisper-small, language=en, independent 20-second audio",
        }
    )
    print("PREPARED", catalog[-1], flush=True)
whisper.unload()
save_segments(ROOT / "segments.whisper.jsonl", segments)
(ROOT / "input-manifest.json").write_text(json.dumps(catalog, indent=2))
print("SAVED", len(segments), "segments", flush=True)
