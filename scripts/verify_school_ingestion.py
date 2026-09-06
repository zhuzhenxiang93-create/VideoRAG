"""Exercise Whisper and OCR without rebuilding the existing news index."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from video_rag.ingestion.media import WhisperTranscriber
from video_rag.ingestion.ocr import PaddleOCRExtractor
from video_rag.storage import load_segments

out = Path("artifacts/tutorialvqa/runtime-regression")
report = json.loads((out / "report.json").read_text())
s = next(
    s for s in load_segments("artifacts/segments.ocr.jsonl") if s.video_id == report["video_id"]
)
for name, run in [
    ("whisper", lambda: WhisperTranscriber().transcribe(s.source_path)),
    ("ocr", lambda: PaddleOCRExtractor(language="ch").extract([s.keyframes[0]])),
    ("ocr_english", lambda: PaddleOCRExtractor(language="en").extract([s.keyframes[0]])),
]:
    try:
        rows = run()
        assert rows, "No output returned"
        (out / f"{name}.json").write_text(
            json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2)
        )
        report[name] = {
            "status": "passed",
            "items": len(rows),
            "input": s.source_path if name == "whisper" else s.keyframes[0].path,
        }
    except Exception as e:
        logging.getLogger(__name__).exception("Ingestion verification failed")
        report[name] = {"status": "failed", "error": repr(e)}
    print(name, report[name], flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
